
from matplotlib import transforms
import torch
import math

import time


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    if (image.shape[0] == 1):
        plt.imshow(image[0], cmap='gray')
    else:
        plt.imshow(reverse_transforms(image))


def train(dataloader, epochs, model, optimizer, device, print_iterations_steps=10000, max_lr=1e-3, min_lr=1e-4, img_size=128):

    warmup_steps = 100
    total_steps = len(dataloader) * epochs

    def lr_lambda_func(step, warmup_steps, total_steps, min_lr, max_lr):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return (min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2) / max_lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda_func(
        step, warmup_steps, total_steps, min_lr, max_lr))
    losses = []
    all_losses = []
    loss_fn = torch.nn.MSELoss()
    model.train()
    best_loss = float('inf')
    step = 0
    start = time.time()
    for epoch in range(epochs):
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            t = torch.randint(model.num_timesteps, (x.shape[0],)).to(device)
            predicted_noise, noise = model(x, t)
            loss = loss_fn(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            scheduler.step()
            losses.append(loss.item())
            all_losses.append(loss.item())
            if (i % print_iterations_steps) == 0:
                end = time.time()
                print(
                    f"Epoch: {epoch}, Iteration: {i}, Loss: {torch.mean(torch.tensor(losses))}, Full Loss:{torch.mean(torch.tensor(all_losses))}, Lr: {scheduler.get_last_lr()[0]}, Time: {end - start}")
                torch.save(model.state_dict(),
                           f"/content/drive/MyDrive/diffusion__best.pth")
                if (i % (print_iterations_steps*10)) == 0:
                    model.eval()
                    with torch.no_grad():
                        results = model.sample(
                            img_size, img_size, model.ending_channels, num_images=1)
                    show_tensor_image(results[0].cpu())
                    plt.show()
                    model.train()
                losses = []
                start = time.time()
        torch.save(model.state_dict(),
                   f"/content/drive/MyDrive/diffusion_{epoch}.pth")
