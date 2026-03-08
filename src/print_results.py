import json

with open('results/metrics_combined.json') as f:
    d = json.load(f)

for model in ['main_model', 'small_model']:
    m = d[model]
    print(f"=== {m['model_id']} ===")
    print("Accuracy:")
    for k,v in m['accuracy'].items():
        print(f"  {k}: {v['mean']:.3f}")
    
    print("\nDrift (last layer):")
    last_layer = list(m['drift']['self_critique_1'].keys())[-1]
    for k,v in m['drift'].items():
        print(f"  {k}: {v[last_layer]['mean']:.4f}")
        
    print("\nProbe Pre vs Post (AUC on last layer):")
    for k in m['probe'].keys():
        try:
            pre_auc = m['probe'][k]['pre'][last_layer]['auc']
            post_auc = m['probe'][k]['post'][last_layer]['auc']
            print(f"  {k}: pre {pre_auc:.3f} -> post {post_auc:.3f}")
        except:
            print(f"  {k}: nan -> nan")
    print()
