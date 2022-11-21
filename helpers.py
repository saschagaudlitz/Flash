import matplotlib.pyplot as plt
import seaborn as sns

def show_marginals(X_train, samples):
    fig,axes = plt.subplots(2,3)
    for i in range(2):
        print(i)
        for j in range(3):
            sns.kdeplot(data = X_train[:,i], fill=True, ax=axes[i,j])
            sns.kdeplot(data = samples[i,:], ax=axes[i,j])
    plt.show()
