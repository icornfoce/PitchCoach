using System;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

namespace PresentationCoach
{
    public enum LLMProvider { OpenAI, Gemini }

    /// <summary>
    /// Turns a finished <see cref="SessionData"/> into personalized coaching text by
    /// calling an LLM. Supports OpenAI Chat Completions and Google Gemini behind one
    /// async API. All network I/O uses UnityWebRequest awaited with async/await
    /// (see the GetAwaiter extension at the bottom) — no coroutines.
    /// </summary>
    public class LLMAdvisorAPI : MonoBehaviour
    {
        [Header("Provider")]
        [SerializeField] private LLMProvider provider = LLMProvider.OpenAI;
        [SerializeField] private string apiKey = "<LLM_API_KEY>";

        [Header("OpenAI")]
        [SerializeField] private string openAiEndpoint = "https://api.openai.com/v1/chat/completions";
        [SerializeField] private string openAiModel = "gpt-4o-mini";

        [Header("Gemini")]
        [SerializeField] private string geminiModel = "gemini-1.5-flash";

        [Header("Generation")]
        [Range(0f, 1f)] [SerializeField] private float temperature = 0.6f;
        [SerializeField] private int timeoutSeconds = 30;

        /// <summary>
        /// The coach persona + output contract. Kept separate from the data so it
        /// can be localized/tuned without touching the payload-building code.
        /// </summary>
        private const string SystemPrompt =
@"You are an expert Thai-speaking presentation coach. You receive objective metrics
from a single practice session (pace, volume, filler words, eye contact, posture)
already scored 0-100. Give warm, specific, actionable feedback IN THAI.

Rules:
- Open with one encouraging sentence grounded in the user's best metric.
- Give 2-3 concrete strengths and 2-3 prioritized improvements, each tied to a
  specific number (e.g. 'พูดเร็วไป 175 คำ/นาที').
- End with ONE short drill they can do next time.
- Be concise: no preamble, no markdown headers, under 180 words total.";

        // ============================================================== //
        //  Public API                                                    //
        // ============================================================== //

        /// <summary>
        /// Sends the session to the configured LLM and returns the feedback text.
        /// Throws on transport/HTTP errors so the caller can show a fallback.
        /// </summary>
        public async Task<string> GetFeedbackAsync(SessionData data)
        {
            data.RecomputeDerived();
            string userMessage = BuildUserMessage(data);

            string url, body;
            if (provider == LLMProvider.OpenAI) BuildOpenAIRequest(userMessage, out url, out body);
            else                                BuildGeminiRequest(userMessage, out url, out body);

            using var req = new UnityWebRequest(url, UnityWebRequest.kHttpVerbPOST);
            req.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
            req.downloadHandler = new DownloadHandlerBuffer();
            req.timeout = timeoutSeconds;
            req.SetRequestHeader("Content-Type", "application/json");

            // Auth differs per provider: OpenAI uses a Bearer header; Gemini takes
            // the key as a query param (added in BuildGeminiRequest).
            if (provider == LLMProvider.OpenAI)
                req.SetRequestHeader("Authorization", $"Bearer {apiKey}");

            await req.SendWebRequest();   // <-- awaitable via extension below

            if (req.result != UnityWebRequest.Result.Success)
                throw new Exception($"LLM request failed [{req.responseCode}]: {req.error}\n{req.downloadHandler.text}");

            return ParseResponse(req.downloadHandler.text);
        }

        // ============================================================== //
        //  Payload construction                                          //
        // ============================================================== //

        /// <summary>Compact, model-friendly summary: raw metrics + 0-100 scores.</summary>
        private string BuildUserMessage(SessionData d)
        {
            var scores = d.GetRadarScores();
            var sb = new StringBuilder();
            sb.AppendLine("Session metrics:");
            sb.AppendLine($"- Duration: {d.totalTimeSeconds:F0}s");
            sb.AppendLine($"- Pace: {d.wordsPerMinute:F0} WPM (score {scores[0]:F0}/100)");
            sb.AppendLine($"- Volume: {d.averageVolumeDb:F1} dBFS (score {scores[1]:F0}/100)");
            sb.AppendLine($"- Filler words: {d.fillerWordCount} (fluency {scores[2]:F0}/100)");
            sb.AppendLine($"- Eye contact: {d.eyeContactPercentage:F0}% (score {scores[3]:F0}/100)");
            sb.AppendLine($"- Posture: {d.postureStatus} ({d.openPosturePercentage:F0}% open, score {scores[4]:F0}/100)");
            sb.AppendLine($"- Overall: {d.OverallScore():F0}/100");
            sb.Append("Write the coaching feedback now.");
            return sb.ToString();
        }

        private void BuildOpenAIRequest(string userMessage, out string url, out string body)
        {
            url = openAiEndpoint;
            var payload = new OpenAIRequest
            {
                model = openAiModel,
                temperature = temperature,
                messages = new[]
                {
                    new OAMessage { role = "system", content = SystemPrompt },
                    new OAMessage { role = "user",   content = userMessage }
                }
            };
            body = JsonUtility.ToJson(payload);
        }

        private void BuildGeminiRequest(string userMessage, out string url, out string body)
        {
            // Gemini authenticates via ?key=, and takes the persona as systemInstruction.
            url = $"https://generativelanguage.googleapis.com/v1beta/models/{geminiModel}:generateContent?key={apiKey}";
            var payload = new GeminiRequest
            {
                systemInstruction = new GeminiContent { parts = new[] { new GeminiPart { text = SystemPrompt } } },
                contents = new[] { new GeminiContent { role = "user", parts = new[] { new GeminiPart { text = userMessage } } } },
                generationConfig = new GeminiGenConfig { temperature = temperature }
            };
            body = JsonUtility.ToJson(payload);
        }

        // ============================================================== //
        //  Response parsing                                              //
        // ============================================================== //

        private string ParseResponse(string json)
        {
            if (provider == LLMProvider.OpenAI)
            {
                var r = JsonUtility.FromJson<OpenAIResponse>(json);
                if (r?.choices != null && r.choices.Length > 0)
                    return r.choices[0].message.content.Trim();
            }
            else
            {
                var r = JsonUtility.FromJson<GeminiResponse>(json);
                if (r?.candidates != null && r.candidates.Length > 0 &&
                    r.candidates[0].content?.parts != null && r.candidates[0].content.parts.Length > 0)
                    return r.candidates[0].content.parts[0].text.Trim();
            }
            throw new Exception($"Could not parse LLM response: {json}");
        }

        // ============================================================== //
        //  DTOs (JsonUtility-compatible: only fields, no dictionaries)    //
        // ============================================================== //

        [Serializable] private class OpenAIRequest { public string model; public float temperature; public OAMessage[] messages; }
        [Serializable] private class OAMessage { public string role; public string content; }
        [Serializable] private class OpenAIResponse { public OAChoice[] choices; }
        [Serializable] private class OAChoice { public OAMessage message; }

        [Serializable] private class GeminiRequest { public GeminiContent systemInstruction; public GeminiContent[] contents; public GeminiGenConfig generationConfig; }
        [Serializable] private class GeminiContent { public string role; public GeminiPart[] parts; }
        [Serializable] private class GeminiPart { public string text; }
        [Serializable] private class GeminiGenConfig { public float temperature; }
        [Serializable] private class GeminiResponse { public GeminiCandidate[] candidates; }
        [Serializable] private class GeminiCandidate { public GeminiContent content; }
    }

    /// <summary>
    /// Lets you `await request.SendWebRequest();` directly. Bridges Unity's
    /// AsyncOperation callback to a TaskAwaiter so the API stays async/await-based.
    /// </summary>
    public static class UnityWebRequestExtensions
    {
        public static TaskAwaiter<UnityWebRequest> GetAwaiter(this UnityWebRequestAsyncOperation op)
        {
            var tcs = new TaskCompletionSource<UnityWebRequest>();
            op.completed += _ => tcs.TrySetResult(op.webRequest);
            return tcs.Task.GetAwaiter();
        }
    }
}
