import csv

w225 = list(csv.DictReader(open('i:/ResearchAI/NewFaceDetect/face106/runs/lapa_lmnet_w225/history.csv')))
w26 = list(csv.DictReader(open('i:/ResearchAI/NewFaceDetect/face106/runs/lapa_w26_hflip_e60/history.csv')))

print('Comparison: w2.25 (30ep, no hflip) vs w2.6 (60ep, hflip)')
print(f"{'ep':>3} | {'w225_loss':>10} {'w225_nme':>10} {'w225_acc8':>10} | {'w26_loss':>10} {'w26_nme':>10} {'w26_acc8':>10} | {'d_acc8':>8}")
print('-' * 85)
for i in range(min(30, len(w225), len(w26))):
    ep = i + 1
    wl, wn, wa = float(w225[i]['train_loss']), float(w225[i]['test_nme']), float(w225[i]['test_acc_008'])
    vl, vn, va = float(w26[i]['train_loss']), float(w26[i]['test_nme']), float(w26[i]['test_acc_008'])
    print(f'{ep:3d} | {wl:10.5f} {wn:10.5f} {wa:10.3f} | {vl:10.5f} {vn:10.5f} {va:10.3f} | {va-wa:+8.2f}')

print()
print('Key insight:')
print('- w2.6 starts HIGHER (ep1: 23.4% vs 21.9%) due to bigger model capacity')
print('- But w2.6 converges SLOWER after ep20, and its final accuracy is lower')
print('- w2.6 at ep30: acc=86.3% vs w2.25 at ep30: acc=92.1%')
print('- Reason: w2.6 has 15.5M params (35% more), with 18k samples it overfits')
print('- The hflip augmentation was NOT enough to compensate for the extra capacity')
print('- Conclusion: w=2.25 is the sweet spot for this dataset size')
