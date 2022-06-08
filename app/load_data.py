import torch
import torchvision
import torchvision.transforms as transforms

# ToTensor：画像のグレースケール化（RGBの0~255を0~1の範囲に正規化）、Normalize：Z値化（RGBの平均と標準偏差を0.5で決め打ちして正規化）
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
gray_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
)


def get_CIFAR_train():
    # トレーニングデータをダウンロード
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )
    return trainloader


def get_CIFAR_test():
    # テストデータをダウンロード
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=True, num_workers=2
    )
    return testloader


def get_MNIST_train():
    # トレーニングデータをダウンロード
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=gray_transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )
    return trainloader


def get_MNIST_test():
    # テストデータをダウンロード
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=gray_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=True, num_workers=2
    )
    return testloader
