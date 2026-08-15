import time

def run_model(model, prompt):
    start = time.perf_counter()

    response = model.invoke(prompt)

    end = time.perf_counter()

    latency = end - start

    return {
        "response": response.content,
        "latency_seconds": round(latency, 3),
        "metadata": response.response_metadata,
    }