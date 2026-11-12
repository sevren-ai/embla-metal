import CryptoKit
import Foundation

/// Models in the Hugging Face cache (`~/.cache/huggingface/hub`, or HF_HUB_CACHE / HF_HOME), in
/// its layout, so a model fetched here or with `hf download` is the same files.
public enum ModelStore {
    public static let defaultRepo = "sevren-ai/Embla1-4bit"
    public static let required = ["config.json", "model.safetensors", "engram_table.safetensors", "tokenizer.json"]

    public static var hubCache: URL {
        let env = ProcessInfo.processInfo.environment
        if let c = env["HF_HUB_CACHE"], !c.isEmpty { return URL(fileURLWithPath: (c as NSString).expandingTildeInPath) }
        return hfHome.appendingPathComponent("hub")
    }

    static var hfHome: URL {
        if let h = ProcessInfo.processInfo.environment["HF_HOME"], !h.isEmpty { return URL(fileURLWithPath: (h as NSString).expandingTildeInPath) }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface")
    }

    static func repoDir(_ repo: String) -> URL {
        hubCache.appendingPathComponent("models--" + repo.replacingOccurrences(of: "/", with: "--"))
    }

    /// A model directory: a path to one, or a repo's snapshot in the cache (its `main`).
    public static func resolve(_ model: String) -> URL? {
        let path = (model as NSString).expandingTildeInPath
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue {
            return URL(fileURLWithPath: path)
        }
        let dir = repoDir(model)
        guard let sha = try? String(contentsOf: dir.appendingPathComponent("refs/main"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines) else { return nil }
        let snap = dir.appendingPathComponent("snapshots/\(sha)")
        return required.allSatisfy { FileManager.default.fileExists(atPath: snap.appendingPathComponent($0).path) } ? snap : nil
    }

    static var token: String? {
        let env = ProcessInfo.processInfo.environment
        for k in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] { if let t = env[k], !t.isEmpty { return t } }
        let file = env["HF_TOKEN_PATH"].map { URL(fileURLWithPath: $0) } ?? hfHome.appendingPathComponent("token")
        return (try? String(contentsOf: file, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func request(_ url: URL) -> URLRequest {
        var r = URLRequest(url: url)
        if let t = token, !t.isEmpty { r.setValue("Bearer \(t)", forHTTPHeaderField: "Authorization") }
        r.setValue("embla-metal", forHTTPHeaderField: "User-Agent")
        return r
    }

    static func getJSON(_ url: URL) throws -> Any {
        final class Box: @unchecked Sendable { var result: Result<(Data, URLResponse), Error>? }
        let sem = DispatchSemaphore(value: 0), box = Box()
        URLSession.shared.dataTask(with: request(url)) { d, r, e in
            box.result = e.map { .failure($0) } ?? .success((d ?? Data(), r!))
            sem.signal()
        }.resume()
        sem.wait()
        let (data, response) = try box.result!.get()
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        if status == 401 || status == 403 || status == 404 {
            throw ChatError.download("\(url.host ?? "") answered \(status): the repo does not exist, or it needs a Hugging Face token (set HF_TOKEN, or run `hf auth login`)")
        }
        guard status == 200 else { throw ChatError.download("\(url) answered \(status)") }
        return try JSONSerialization.jsonObject(with: data)
    }

    /// Downloads `repo`'s files at `main` into the cache (resuming partial files, checking the
    /// large ones' SHA-256); returns the snapshot directory.
    @discardableResult
    public static func download(_ repo: String, log: (String) -> Void = { print($0) }) throws -> URL {
        let api = URL(string: "https://huggingface.co/api/models/\(repo)/revision/main")!
        guard let info = try getJSON(api) as? [String: Any], let sha = info["sha"] as? String else {
            throw ChatError.download("no revision for \(repo)")
        }
        let tree = URL(string: "https://huggingface.co/api/models/\(repo)/tree/\(sha)?recursive=true")!
        let entries = (try getJSON(tree) as? [[String: Any]] ?? []).filter { $0["type"] as? String == "file" }
        let dir = repoDir(repo)
        let fm = FileManager.default
        let snap = dir.appendingPathComponent("snapshots/\(sha)")
        try fm.createDirectory(at: dir.appendingPathComponent("blobs"), withIntermediateDirectories: true)
        try fm.createDirectory(at: snap, withIntermediateDirectories: true)
        let total = entries.reduce(0) { $0 + (($1["lfs"] as? [String: Any])?["size"] as? Int ?? $1["size"] as? Int ?? 0) }
        log(String(format: "%@ at %@: %d files, %.1f GB", repo, String(sha.prefix(8)), entries.count, Double(total) / 1e9))
        for e in entries {
            guard let path = e["path"] as? String, let oid = e["oid"] as? String else { continue }
            let lfs = e["lfs"] as? [String: Any]
            let blobName = lfs?["oid"] as? String ?? oid
            let size = lfs?["size"] as? Int ?? e["size"] as? Int ?? 0
            let blob = dir.appendingPathComponent("blobs/\(blobName)")
            if (try? fm.attributesOfItem(atPath: blob.path)[.size] as? Int) == size {
                log("  \(path): have it")
            } else {
                let url = URL(string: "https://huggingface.co/\(repo)/resolve/\(sha)/\(path)")!
                try fetch(url, to: blob, size: size, sha256: lfs?["oid"] as? String, name: path, log: log)
            }
            let link = snap.appendingPathComponent(path)
            try fm.createDirectory(at: link.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? fm.removeItem(at: link)
            let depth = path.split(separator: "/").count - 1
            try fm.createSymbolicLink(atPath: link.path, withDestinationPath: String(repeating: "../", count: depth + 2) + "blobs/\(blobName)")
        }
        try fm.createDirectory(at: dir.appendingPathComponent("refs"), withIntermediateDirectories: true)
        try sha.write(to: dir.appendingPathComponent("refs/main"), atomically: true, encoding: .utf8)
        return snap
    }

    /// One file to `dest`, through `dest.incomplete` (resumed from its length).
    static func fetch(_ url: URL, to dest: URL, size: Int, sha256: String?, name: String, log: (String) -> Void) throws {
        let part = dest.appendingPathExtension("incomplete")
        let fm = FileManager.default
        if !fm.fileExists(atPath: part.path) { fm.createFile(atPath: part.path, contents: nil) }
        var have = (try fm.attributesOfItem(atPath: part.path)[.size] as? Int) ?? 0
        if have > size { try Data().write(to: part); have = 0 }
        if have < size {
            var req = request(url)
            if have > 0 { req.setValue("bytes=\(have)-", forHTTPHeaderField: "Range") }
            let loader = Loader(file: try FileHandle(forWritingTo: part), offset: have, size: size, name: name)
            let session = URLSession(configuration: .default, delegate: loader, delegateQueue: nil)
            session.dataTask(with: req).resume()
            loader.done.wait()
            session.invalidateAndCancel()
            if let err = loader.error { throw ChatError.download("\(name): \(err)") }
        }
        if let want = sha256 {
            log("  \(name): checking")
            var h = SHA256()
            let fh = try FileHandle(forReadingFrom: part)
            while let chunk = try fh.read(upToCount: 8 << 20), !chunk.isEmpty { h.update(data: chunk) }
            try fh.close()
            let got = h.finalize().map { String(format: "%02x", $0) }.joined()
            guard got == want else {
                try? fm.removeItem(at: part)
                throw ChatError.download("\(name): SHA-256 mismatch (removed it; run again)")
            }
        }
        try? fm.removeItem(at: dest)
        try fm.moveItem(at: part, to: dest)
        log("  \(name): done")
    }

    final class Loader: NSObject, URLSessionDataDelegate, @unchecked Sendable {
        let file: FileHandle, size: Int, name: String
        var got: Int, error: String?
        let done = DispatchSemaphore(value: 0)
        let start = Date(), startBytes: Int
        var lastPrint = Date.distantPast

        init(file: FileHandle, offset: Int, size: Int, name: String) {
            self.file = file
            self.got = offset
            self.startBytes = offset
            self.size = size
            self.name = name
            super.init()
        }

        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            if status == 200 && got > 0 {
                // no range support: start over
                _ = try? file.truncate(atOffset: 0)
                got = 0
            } else if status != 200 && status != 206 {
                error = "HTTP \(status)" + (status == 401 ? " (needs a Hugging Face token: set HF_TOKEN, or run `hf auth login`)" : "")
                completionHandler(.cancel)
                return
            }
            _ = try? file.seekToEnd()
            completionHandler(.allow)
        }

        /// The token goes to Hugging Face only, not to the storage a file redirects to.
        func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                        newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
            var r = request
            if r.url?.host != "huggingface.co" { r.setValue(nil, forHTTPHeaderField: "Authorization") }
            completionHandler(r)
        }

        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
            do { try file.write(contentsOf: data) } catch { self.error = "\(error)"; dataTask.cancel(); return }
            got += data.count
            if Date().timeIntervalSince(lastPrint) > 0.5 || got == size {
                lastPrint = Date()
                let rate = Double(got - startBytes) / max(0.001, Date().timeIntervalSince(start)) / 1e6
                let line = String(format: "\r  %@: %.2f / %.2f GB (%.0f MB/s)  ", name, Double(got) / 1e9, Double(size) / 1e9, rate)
                FileHandle.standardError.write(line.data(using: .utf8)!)
            }
        }

        func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
            if size > 1 << 20 { FileHandle.standardError.write("\n".data(using: .utf8)!) }
            if let error, self.error == nil { self.error = error.localizedDescription }
            if self.error == nil && got != size { self.error = "got \(got) of \(size) bytes (run again to resume)" }
            try? file.close()
            done.signal()
        }
    }
}
