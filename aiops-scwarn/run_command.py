import subprocess

def main():
    command = [
        "python3", "-m", "run_yunzhanghu",
        "--publish_date", "20240627",
        "--prometheus_address", "172.16.17.114:19192",
        "--service", "yid_k8s",
        "--sc_id", "10045",
        "--train_date", "2024-06-21 16:33:14",
        "--task_count", "5",
        "--step", "120",
        "--timeout", "10000",
        "--train_duration", "432000",
        "--detection_duration", "3600",
        "--predict_interval", "30"
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Command executed successfully.")
        print(result.stdout)
    else:
        print("Command failed with return code", result.returncode)
        print(result.stderr)

if __name__ == "__main__":
    main()
