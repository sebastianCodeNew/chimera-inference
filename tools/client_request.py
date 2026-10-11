"""Client test script sending real HTTP request to the running Chimera-Inference server."""

import json
import urllib.request


def main() -> None:
    print("Sending request to http://localhost:8000/v1/chat/completions...")

    payload = {
        "model": "qwen2.5-1.5b-instruct",
        "messages": [
            {
                "role": "user",
                "content": "Halo! Berikan 1 tips produktivitas terbaik untuk software engineer.",
            }
        ],
        "max_tokens": 40,
        "temperature": 0.7,
        "min_p": 0.05,
    }

    req = urllib.request.Request(
        "http://localhost:8000/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print("\n[HTTP Response Status]:", resp.status)
        print("[Model Answer]:")
        print("--------------------------------------------------")
        print(data["choices"][0]["message"]["content"])
        print("--------------------------------------------------")
        print("[Finish Reason]:", data["choices"][0]["finish_reason"])
        print("[Token Usage]:", data.get("usage"))


if __name__ == "__main__":
    main()
