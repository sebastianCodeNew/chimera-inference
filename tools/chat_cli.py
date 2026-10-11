"""Interactive CLI Chat terminal connected to Chimera-Inference server."""

import json
import sys
import urllib.request


def main() -> None:
    print("=" * 60)
    print("Chimera-Inference: Interactive Terminal Chat")
    print("Connected to: http://localhost:8000 (NVIDIA RTX 3050 CUDA)")
    print("Type 'exit' or 'quit' to end session.")
    print("=" * 60)

    # Check connection
    try:
        with urllib.request.urlopen("http://localhost:8000/health", timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(
                f"Server Status: OK | Device: {data['device']} | VRAM: {data.get('vram_allocated_mb', 0):.0f} MiB\n"
            )
    except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as e:
        print(f"Error connecting to server: {e}", file=sys.stderr)
        return

    history = []

    while True:
        try:
            user_input = input("\n[Anda]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSampai jumpa!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Sesi chat selesai. Sampai jumpa!")
            break

        history.append({"role": "user", "content": user_input})

        payload = {
            "model": "qwen2.5-1.5b-instruct",
            "messages": history,
            "max_tokens": 256,
            "temperature": 0.7,
            "min_p": 0.05,
            "stream": True,
        }

        req = urllib.request.Request(
            "http://localhost:8000/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        sys.stdout.write("[Qwen]: ")
        sys.stdout.flush()

        ai_response = []
        try:
            with urllib.request.urlopen(req) as resp:
                for line_bytes in resp:
                    line = line_bytes.decode("utf-8").strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            sys.stdout.write(delta)
                            sys.stdout.flush()
                            ai_response.append(delta)
            sys.stdout.write("\n")
            history.append({"role": "assistant", "content": "".join(ai_response)})
        except (
            urllib.error.URLError,
            OSError,
            TimeoutError,
            json.JSONDecodeError,
        ) as e:
            print(f"\n[Error]: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
