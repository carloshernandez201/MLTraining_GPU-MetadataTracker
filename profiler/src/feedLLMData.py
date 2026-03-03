import time
import json
import openai


def parse_jsonl(path, interval: int):
    with open(path, "rb") as read_file:
        read_file.seek(0, 2)
        pos = read_file.tell()
        cutoff = time.time() - interval
        curtime = time.time()
        lines = []

        # pytorch stats (reading backwards so first hit = newest)
        loss_newest = 0
        loss_oldest = 0
        step_newest = 0
        step_oldest = 0
        found_pytorch = False

        while curtime >= cutoff:
            localbytect = pos - 1
            while localbytect > 0:
                read_file.seek(localbytect)
                if read_file.read(1) == b"\n":
                    break
                localbytect -= 1

            # read from newline to previous pos
            read_file.seek(localbytect + 1 if localbytect > 0 else 0)
            raw = read_file.read(pos - (localbytect + 1 if localbytect > 0 else 0))
            pos = localbytect
            if pos <= 0:
                break

            try:
                entry = json.loads(raw.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            curtime = entry.get("timestamp", 0)
            if curtime < cutoff:
                break

            if entry.get("source") == "pytorch":
                if not found_pytorch:
                    loss_newest = entry["data"]["loss"]
                    step_newest = entry["data"]["step"]
                    found_pytorch = True
                loss_oldest = entry["data"]["loss"]
                step_oldest = entry["data"]["step"]
            else:
                lines.append(entry)

        # add pytorch summary as one entry
        if found_pytorch:
            lines.append({
                "source": "pytorch_summary",
                "data": {
                    "loss_newest": loss_newest,
                    "loss_oldest": loss_oldest,
                    "step_newest": step_newest,
                    "step_oldest": step_oldest,
                    "loss_trend": "decreasing" if loss_newest < loss_oldest else "increasing" if loss_newest > loss_oldest else "flat",
                }
            })

    gpu_stats = [l for l in lines if l.get("source") == "gpu"]
    pytorch_summary = next((l for l in lines if l.get("source") == "pytorch_summary"), {})
    slurm_stats = [l for l in lines if l.get("source") == "slurm"]

    return json.dumps({
        "gpu": gpu_stats,
        "pytorch": pytorch_summary,
        "slurm": slurm_stats,
    })


def get_ai_response(path, interval=60):
    message = parse_jsonl(path, interval)

    client = openai.OpenAI()  # reads OPENAI_API_KEY from env

    messages = [
        {
            "role": "system",
            "content": (
                "You are part of an app that analyzes ML training process data including "
                "gpu data, pytorch loop data, and slurm data to analyze whether training "
                "hyperparameters are set efficiently. You recommend changes to make training "
                "more safe and efficient by modifying the system or pytorch loop. "
                "Respond ONLY with a JSON object with keys 'action' and 'reason'."
            ),
        },
        {"role": "user", "content": message},
    ]

    chat = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.1)
    reply = chat.choices[0].message.content
    with open("control_plane.json", 'w') as outfile:
        outfile.write(reply)
    