def calculate_far_frr(results):
    false_accepts = 0
    false_rejects = 0
    total_imposters = 0
    total_genuine = 0

    for actual, predicted in results:
        if actual == predicted:
            total_genuine += 1
        else:
            if predicted != "Unknown":
                false_accepts += 1
                total_imposters += 1
            else:
                false_rejects += 1
                total_genuine += 1

    FAR = false_accepts / max(total_imposters, 1)
    FRR = false_rejects / max(total_genuine, 1)

    return FAR, FRR

results = [
    ("Yuvika", "Yuvika"),
    ("Yuvika", "Unknown"),
    ("Friend", "Yuvika"),
    ("Unknown", "Unknown")
]

far, frr = calculate_far_frr(results)

print("FAR:", far)
print("FRR:", frr)