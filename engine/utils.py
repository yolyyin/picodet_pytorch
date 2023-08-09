import torch
import matplotlib.pyplot as plt


def save_loss_plot(name, output_dir, train_loss):
    figure, train_ax = plt.subplots()
    train_ax.plot(train_loss, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    figure.savefig(f"{output_dir}/{name}.png")
    print(f"Saving plots to {output_dir}...")
    plt.close('all')


def save_model(name, output_dir, epoch,model, optimizer):
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"{output_dir}/{name}.pth")


def resume_paras(model,
                 para_path='D:/2023_det/output/baseline/baseline.state'):
    state_dict = torch.load(para_path)
    model.load_state_dict(state_dict)


def resume_state(model,optimizer, para_path):
    model_state_dict = torch.load(para_path)['model_state_dict']
    model.load_state_dict(model_state_dict)
    opt_dict = torch.load(para_path)['optimizer_state_dict']
    optimizer.load_state_dict(opt_dict)