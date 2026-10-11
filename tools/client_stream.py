"""Client script testing real-time SSE streaming from the running server."""

import json
import sys
import urllib.request


def main() -> None:
    print("Connecting to http://localhost:8000/v1/chat/completions (SSE Stream)...")

    payload = {
        "model": "qwen2.5-1.5b-instruct",
        "messages": [
            {
                "role": "user",
                "content": "Jelaskan konsep dasar pemrograman berorientasi objek (OOP) dalam 2 kalimat ringkas.",
            }
        ],
        "max_tokens": 50,
        "temperature": 0.7,
        "min_p": 0.05,
        "stream": True,
    }

    req = urllib.request.Request(
        "http://localhost:8000/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    print("\n[Streaming Output Real-time]:")
    print("--------------------------------------------------")
    with urllib.request.urlopen(req) as resp:
        for line_bytes in resp:
            line = line_bytes.decode("utf-8").strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    sys.stdout.write(delta)
                    sys.stdout.flush()
    print("\n--------------------------------------------------")
    print("\nStream completed successfully!")


if __name__ == "__main__":
    main()
