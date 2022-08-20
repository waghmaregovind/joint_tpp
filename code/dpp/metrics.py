from sklearn import metrics
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import dpp
sns.set_style('whitegrid')


def mark_metrics(lengths, mark_pred, mark_gt):
    mark_pr_f = []
    mark_gt_f = []
    for i in range(len(lengths)):
        for j in range(len(lengths[i])):
            for k in range(lengths[i][j]):
                mark_pr_f.append(mark_pred[i][j][k].detach().cpu())
                mark_gt_f.append(mark_gt[i][j][k].detach().cpu())
                
    acc = metrics.accuracy_score(mark_gt_f, mark_pr_f) * 100
    f1_micro = metrics.f1_score(mark_gt_f, mark_pr_f, average='micro') * 100
    f1_macro = metrics.f1_score(mark_gt_f, mark_pr_f, average='macro') * 100
    f1_weighted = metrics.f1_score(mark_gt_f, mark_pr_f, average='weighted') * 100    
    
    print(f"f1_micro (Acc) : {f1_micro:.4f}")
    print(f"f1_macro       : {f1_macro:.4f}")
    print(f"f1_weighted    : {f1_weighted:.4f}")

def nll_metrics(time_nll, mark_nll, total_nll, total_count, interval):
    
    time_nll = np.concatenate(time_nll)
    mark_nll = np.concatenate(mark_nll)
    total_nll = np.concatenate(total_nll)
    interval = np.concatenate(interval)
    
    total_nll_by_time = total_nll / interval
    
    tot_time_nll = time_nll.sum() / total_count
    tot_mark_nll = mark_nll.sum() / total_count
    tot_nll = total_nll.sum() / total_count
    tot_nll_by_time = total_nll_by_time.sum() / total_count
    
    print(f"Time_NLL       : {tot_time_nll:.4f}")
    print(f"Mark_NLL       : {tot_mark_nll:.4f}")
    print(f"NLL            : {tot_nll:.4f}")
    print(f"NLL/TIME       : {tot_nll_by_time:.4f}")    

def aggregate_loss_over_dataloader(model, dl, eval_mode=False):
    
    if eval_mode:
        total_loss = 0.0
        total_count = 0
        
        time_nll = []
        mark_nll = []
        surv_nll = []
        total_nll = []
        
        lengths = []
        mark_pred = []
        mark_gt = []
        
        interval = []
        
        with torch.no_grad():
            for batch in dl:
                tot_nll, log_p, log_mark, log_surv, mark_pred_batch = model.log_prob(batch)
                total_loss += (-1)*tot_nll.sum().item()
                total_count += batch.size
                
                time_nll.append(-log_p.sum(-1).detach().cpu().numpy()) #(batch_size,)
                mark_nll.append(-log_mark.sum(-1).detach().cpu().numpy()) #(batch_size,)
                total_nll.append(-tot_nll.detach().cpu().numpy()) #(batch_size,)
                surv_nll.append(-log_surv.detach().cpu().numpy()) #(batch_size,)
                 
                lengths.append(batch.mask.sum(-1).int())
                mark_pred.append(mark_pred_batch)
                mark_gt.append(batch.marks)
                
                interval.append(batch.interval.detach().cpu().numpy())
            
            nll_metrics(time_nll, mark_nll, total_nll, total_count, interval)
            mark_metrics(lengths, mark_pred, mark_gt)
            
    else:
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for batch in dl:
                tot_nll, log_p, log_mark, log_surv, mark_pred = model.log_prob(batch)
                total_loss += (-1)*tot_nll.sum().item()
                total_count += batch.size
                
    return total_loss / total_count


# Code from https://colab.research.google.com/github/shchur/shchur.github.io/blob/gh-pages/assets/notebooks/tpp2/neural_tpp.ipynb
def sampling_plot(model, t_end, num_seq, dataset):
    with torch.no_grad():
        sampled_batch = model.sample(t_end=t_end, batch_size=num_seq)
        real_batch = dpp.data.Batch.from_list([s for s in dataset])  

        fig, axes = plt.subplots(figsize=[8, 4.5], dpi=200, nrows=2, ncols=2)

        for idx, t in enumerate(real_batch.inter_times.cumsum(-1).cpu().numpy()[:8]):
            axes[0,0].scatter(t, np.ones_like(t) * idx, alpha=0.5, c='C2', marker="|")
        axes[0,0].set_title("Arrival times: Real event sequences", fontsize=7)
        axes[0,0].set_xlabel("Time", fontsize=7)
        axes[0,0].set_ylabel("Different event sequences", fontsize=7)
        axes[0,0].set_yticks(np.arange(8));
        axes[0,0].xaxis.offsetText.set_fontsize(7)

        for idx, t in enumerate(sampled_batch.inter_times.cumsum(-1).cpu().numpy()[:8]):
            axes[0,1].scatter(t, np.ones_like(t) * idx, alpha=0.5, c='C3', marker="|")
        axes[0,1].set_xlabel("Time", fontsize=7)
        axes[0,1].set_title("Arrival times: Sampled event sequences", fontsize=7)
        axes[0,1].set_ylabel("Different event sequences", fontsize=7)
        axes[0,1].set_yticks(np.arange(8))
        axes[0,1].xaxis.offsetText.set_fontsize(7)

        sample_len = sampled_batch.mask.sum(-1).cpu().numpy()
        real_len = real_batch.mask.sum(-1).cpu().numpy()
        
        axes[1,0].set_title("Distribution of sequence lengths", fontsize=7)
        q_min = min(real_len.min(), sample_len.min()).astype(int)
        q_max = max(real_len.max(), sample_len.max()).astype(int)
        axes[1,0].hist([real_len, sample_len], bins=30, alpha=0.9, color=["C2","C3"], range=(q_min, q_max), label=["Real data", "Sampled data"]);
        axes[1,0].set_xlabel(r"Sequence length", fontsize=7)
        axes[1,0].set_ylabel("Frequency", fontsize=7)
        
        sampled_marks_flat = []
        real_marks_flat = []
        
        for i, each in enumerate(sampled_batch.marks):
            sampled_marks_flat.append(sampled_batch.marks[i, :sampled_batch.mask[i].sum().int()].detach().cpu().numpy())
        
        for i, each in enumerate(real_batch.marks):
            real_marks_flat.append(real_batch.marks[i, :real_batch.mask[i].sum().int()].detach().cpu().numpy())

        sampled_marks_flat = np.concatenate(sampled_marks_flat)
        real_marks_flat = np.concatenate(real_marks_flat)

        axes[1,1].set_title("Distribution of marks", fontsize=7)
        unique, counts = np.unique(np.asarray(sampled_marks_flat), return_counts=True)
        unique_0, counts_0 = np.unique(np.asarray(real_marks_flat), return_counts=True)
        
        q_min = min(unique.min(), unique_0.min()).astype(int)
        q_max = max(unique.max(), unique_0.max()).astype(int)
        axes[1,1].hist([real_marks_flat, sampled_marks_flat], alpha=0.9, color=["C2","C3"], range=(q_min, q_max), label=["Real data", "Sampled data"]);
        axes[1,1].set_xlabel(r"Marks", fontsize=7)
        axes[1,1].set_ylabel("Frequency", fontsize=7)

        axes[1,0].legend(ncol=1, fontsize=7)
        axes[1,1].legend(ncol=1, fontsize=7)

        axes[1,1].yaxis.offsetText.set_fontsize(7)

        for ax in np.ravel(axes):
            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=7)

        fig.tight_layout()