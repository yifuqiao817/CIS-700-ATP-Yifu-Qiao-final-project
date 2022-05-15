import matplotlib.pyplot as plt
import numpy as np
import torch


hyper_para = {
  'epochs': 32,
  'M': .4, # short
  'N': .6, # long
  'max_len': 512,
  'dropout': 0.5,
  'batch_size': 32,
  'use_tokens': False,
  'verbose': 1,
  'lr': 0.00005
}

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 8))

for i in range(3):
    t_loss_short =  torch.load('./loss/t_loss_short_rep{}.pt'.format(i))
    v_loss_short =  torch.load('./loss/v_loss_short_rep{}.pt'.format(i))
    t_loss_long =  torch.load('./loss/t_loss_long_rep{}.pt'.format(i))
    v_loss_long =  torch.load('./loss/v_loss_long_rep{}.pt'.format(i))
    test_loss = torch.load('./loss/test_loss_rep{}.pt'.format(i))

    # short, training
    plt.subplot(2, 3, 1)
    t_loss_short= [x.to("cpu") for x in t_loss_short]
    # plt.scatter(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in t_loss_short])
    plt.plot(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in t_loss_short])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training (short, rep {})'.format(i+1))

    # short, validation
    plt.subplot(2, 3, 2)
    v_loss_short= [x.to("cpu") for x in v_loss_short]
    # plt.scatter(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in v_loss_short])
    plt.plot(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in v_loss_short])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('validation (short, rep {})'.format(i+1))

    # long, training
    plt.subplot(2, 3, 3)
    t_loss_long= [x.to("cpu") for x in t_loss_long]
    # plt.scatter(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in t_loss_long])
    plt.plot(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in t_loss_long])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training (long, rep {})'.format(i+1))

    # long, validation
    plt.subplot(2, 3, 4)
    v_loss_long= [x.to("cpu") for x in v_loss_long]
    # plt.scatter(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in v_loss_long])
    plt.plot(np.arange(hyper_para['epochs']) + 1, [l.detach().numpy() for l in v_loss_long])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('validation (long, rep {})'.format(i+1))

    # test
    plt.subplot(2, 3, 5)
    # plt.scatter(np.arange(len(test_loss)) + 1, [l.detach().numpy() for l in test_loss])
    plt.plot(np.arange(len(test_loss)) + 1, [l.detach().numpy() for l in test_loss])
    plt.xlabel('the number of validation samples')
    plt.ylabel('loss')
    plt.title('test, rep {}'.format(i+1))

plt.tight_layout()
plt.show()
# plt.savefig('./plots/results.png')
plt.close()
