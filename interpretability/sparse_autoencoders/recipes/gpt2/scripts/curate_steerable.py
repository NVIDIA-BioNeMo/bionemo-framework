import json
import urllib.request


BASE = "http://localhost:8749"


def post(p, b):
    """POST a JSON body to the running dashboard API and return the parsed response."""
    r = urllib.request.Request(BASE + p, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=120))


# Diverse concept probes: a natural sentence that should light up the concept's SAE feature.
probes = {
    "dog": "The loyal dog wagged its tail and barked at the mailman.",
    "cat": "The cat sat by the window and purred softly.",
    "France": "We visited Paris and admired the Eiffel Tower in France.",
    "London": "Big Ben and the London Underground were crowded with tourists.",
    "ocean": "The waves crashed on the beach as the ocean tide came in.",
    "mountain": "They climbed the steep mountain peak covered in snow.",
    "anger": "He was absolutely furious and shouted in a rage.",
    "happiness": "She smiled with pure joy and laughed happily all day.",
    "sadness": "She was crying with grief and sorrow after the loss.",
    "fear": "He trembled in terror, afraid of the dark shadows.",
    "love": "They fell deeply in love and cherished each other dearly.",
    "music": "The guitar and drums filled the concert hall with music.",
    "food": "The delicious dinner was a tasty meal of roasted chicken.",
    "football": "The quarterback threw a touchdown pass to win the football game.",
    "space": "The rocket launched toward the distant stars and planets.",
    "war": "The soldiers fought fierce battles on the war front.",
    "medicine": "The doctor prescribed medicine to treat the patient's illness.",
    "law": "The lawyer argued the case before the judge in court.",
    "religion": "The priest prayed in the church and blessed the congregation.",
    "money": "She counted the cash and deposited the money in the bank.",
    "computers": "The programmer wrote software code and debugged the computer.",
    "weather": "The storm brought heavy rain, thunder and strong winds.",
    "science": "The scientist ran an experiment and measured the results.",
    "art": "The painter mixed colors on the canvas to create art.",
    "medieval": "The knight drew his sword and defended the castle.",
    "politics": "The senator debated the new policy during the election campaign.",
    "cooking": "She chopped onions and simmered the sauce on the stove.",
    "cars": "The engine roared as the sports car sped down the highway.",
    "school": "The students studied for their exam in the classroom.",
    "nature": "Birds sang in the green forest among the tall trees.",
}

PROMPT = "I think the best thing to do is"
N, STR = 24, 40
base = post("/api/generate", {"prompt": PROMPT, "n_tokens": N, "temperature": 0})["generation"]["sequence"]
print("prompt : %r  (strength %d, greedy)" % (PROMPT, STR))
print("BASELINE: %r\n" % base[len(PROMPT) :])

results = []
for name, probe in probes.items():
    ann = post("/api/annotate", {"sequence": probe, "mode": "topk", "k": 1})
    f = ann["features"][0]
    fid, peak, lab = f["feature_id"], f["max_activation"], f["label"]
    g = post(
        "/api/generate",
        {"prompt": PROMPT, "features": [{"feature_id": fid, "strength": STR}], "n_tokens": N, "temperature": 0},
    )
    steered = g["generation"]["sequence"][len(PROMPT) :]
    results.append({"concept": name, "feature_id": fid, "peak": peak, "np_label": lab, "steered": steered})
    print("[%-10s] #%-6d peak %4.0f | %s" % (name, fid, peak, lab))
    print("            -> %r\n" % steered)

json.dump(results, open("/workspace/m0-gpt2/steerable_candidates.json", "w"), indent=1)
print("wrote steerable_candidates.json (%d candidates)" % len(results))
