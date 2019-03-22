import matplotlib.pyplot as plt
import numpy as np

def plot_mean(mean, labels, start_index=1):
    fig, ax = plt.subplots(figsize=(9,6))
    x_dim = list(range(start_index,start_index+mean.shape[0]))
    for i in range(mean.shape[1]):
        ax.plot(x_dim, mean[:,i] , label=labels[i])
    title = "Log marginal likelihood vs. rank in trigram data"
    ax.set_title(title, size=15)
    ax.set_ylabel("Log marginal likelihood",size=13)
    ax.set_xlabel("Rank",size=12)
    ax.legend()
    fig.savefig(title.replace(" ", "_").lower()+".pdf",  format="pdf")
    
def plot_variance(mean, variance):
    fig, ax = plt.subplots(figsize=(9,6))
    x_dim = list(range(1,102,5))
    ax.plot(x_dim, 2*variance[:,0] - 2*mean[:,0] , label="SIS")
    ax.plot(x_dim, 2*variance[:,1] - 2*mean[:,1] , label="ARS")
    ax.plot(x_dim, 2*variance[:,2] - 2*mean[:,2], label="PARS")
    ax.plot(x_dim, 2*variance[:,3] - 2*mean[:,3], label="VB")
    ax.set_title("Relative variance vs. Rank")
    ax.set_ylabel("Relative variance")
    ax.set_xlabel("Rank")
    ax.legend()
    
def plot_mean_with_variance_around(mean, variance):
    fig, ax = plt.subplots(figsize=(9,6))
    x_dim = list(range(1,102,5))
    ax.plot(x_dim, mean[:,0] , label="SIS")
    upper_0 = - 2 * variance[:,0] + 3*mean[:,0]
    lower_0 = + 2*variance[:,0] - mean[:,0]
    ax.fill_between(x_dim, upper_0, lower_0, alpha=0.15)
    ax.plot(x_dim, mean[:,1] , label="APF")
    upper_1 = - 2 * variance[:,1] + 3*mean[:,1]
    lower_1 = + 2*variance[:,1] - mean[:,1]
    ax.fill_between(x_dim, upper_1, lower_1, alpha=0.15)
    ax.plot(x_dim, mean[:,2], label="Patient APF")
    upper_2 = - 2 * variance[:,2] + 3*mean[:,2]
    lower_2 = + 2*variance[:,2] - mean[:,2]
    ax.fill_between(x_dim, upper_2, lower_2, alpha=0.15)
    ax.plot(x_dim, mean[:,3], label="VB")
    upper_3 = - 2 * variance[:,3] + 3*mean[:,3]
    lower_3 = + 2*variance[:,3] - mean[:,3]
    ax.fill_between(x_dim, upper_3, lower_3, alpha=0.15)
    ax.set_title("Log marginal likelihood vs. Rank")
    ax.set_ylabel("Log marginal likelihood")
    ax.set_xlabel("Rank")
    ax.legend()
    
def plot_ess(ess):
    fig, ax = plt.subplots(figsize=(9,6))
    x_dim = list(range(1,102,5))
    ax.plot(x_dim, ess[:,0], label="SIS")
    ax.plot(x_dim, ess[:,1], label="ARS")
    ax.plot(x_dim, ess[:,2], label="PARS")
    ax.set_title("ESS vs. Rank")
    ax.set_ylabel("ESS")
    ax.set_xlabel("Rank")
    ax.legend()
    
desired_letter_order = 'aeioubcdfghjklmnpqrstvwxyz'

def convert_to_desired_order(S_IR, S_JR, S_KR):
    ir = pd.DataFrame(S_IR, index=list(string.ascii_lowercase)).loc[list(desired_letter_order)]
    jr = pd.DataFrame(S_JR, index=list(string.ascii_lowercase)).loc[list(desired_letter_order)]
    kr = pd.DataFrame(S_KR, index=list(string.ascii_lowercase)).loc[list(desired_letter_order)]
    return ir, jr, kr
    
def plot_rank(S, save=False, title=""):
    S_R, S_IR, S_JR, S_KR = S
    ir, jr, kr = convert_to_desired_order(S_IR, S_JR, S_KR)
    r = S_IR.shape[1]
    fig, axes = plt.subplots(1,3, figsize=(10,8));

    axes[0].imshow(ir)
    axes[1].imshow(jr)
    axes[2].imshow(kr)

    axes[0].set_yticks(range(26))
    axes[0].set_yticklabels(desired_letter_order)
    axes[1].set_yticks(range(26))
    axes[1].set_yticklabels(desired_letter_order)
    axes[2].set_yticks(range(26))
    axes[2].set_yticklabels(desired_letter_order)

    axes[0].set_xticks(range(r))
    axes[0].set_xticklabels(range(1,r+1))
    axes[1].set_xticks(range(r))
    axes[1].set_xticklabels(range(1,r+1))
    axes[2].set_xticks(range(r))
    axes[2].set_xticklabels(range(1,r+1))

    axes[0].set_title("$X_1$")
    axes[1].set_title("$X_2$")
    axes[2].set_title("$X_3$")
    fig.tight_layout(pad=3, rect=[0.12, .0, .88, 1.])
    if title != "":
        fig.suptitle(title, size=14)
    if save==True:
        fig.savefig(title.replace(" ", "_").lower()+".pdf",  format="pdf", bbox_inches='tight')
    return S_R, S_IR, S_JR, S_KR


#test this
def cumsum_max(array, limit=0.95):
    array = array / array.sum()
    arg_indices_sorted = np.argsort(-array) 
    cumsums = np.cumsum(array[arg_indices_sorted])
    for i in range(len(array)):
        if cumsums[i] > limit:
            return np.sort(arg_indices_sorted[:i+1])

#test this
def get_bundles(S_IR, S_JR, S_KR, limit=0.95):
    bundles = []
    R = S_IR.shape[1]
    for r in range(R):
        bundle = []
        bundle.append(cumsum_max(S_IR[:, r], limit))
        bundle.append(cumsum_max(S_JR[:, r], limit))
        bundle.append(cumsum_max(S_KR[:, r], limit))
        bundle = np.array(bundle)
        bundles.append(bundle)    
    return bundles

def print_transitions(S_IR, S_JR, S_KR, limit=0.95):
    bundles = get_bundles(S_IR, S_JR, S_KR, limit)
    for r, bundle in enumerate(bundles):
        print("for rank {}".format(r+1))
        print(list(map(lambda x: [c for c in string.ascii_lowercase][x], bundle[0])))
        print("->")
        print(list(map(lambda x: [c for c in string.ascii_lowercase][x], bundle[1])))
        print("->")
        print(list(map(lambda x: [c for c in string.ascii_lowercase][x], bundle[2]))) 
       