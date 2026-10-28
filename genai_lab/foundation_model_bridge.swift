import Foundation
import FoundationModels

struct BridgeRequest: Decodable {
    let instructions: String
    let prompt: String
}

struct BridgeResponse: Encodable {
    let content: String?
    let error: String?
}

@main
struct FoundationModelBridge {
    static func main() async {
        do {
            let input = FileHandle.standardInput.readDataToEndOfFile()
            let request = try JSONDecoder().decode(BridgeRequest.self, from: input)

            guard case .available = SystemLanguageModel.default.availability else {
                throw NSError(
                    domain: "MovieBanditFoundationModel",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Apple Foundation Models is unavailable on this Mac."]
                )
            }

            let session = LanguageModelSession(instructions: request.instructions)
            let response = try await session.respond(to: request.prompt)
            try write(BridgeResponse(content: response.content, error: nil))
        } catch {
            try? write(BridgeResponse(content: nil, error: String(describing: error)))
        }
    }

    private static func write(_ response: BridgeResponse) throws {
        let data = try JSONEncoder().encode(response)
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data([0x0A]))
    }
}
