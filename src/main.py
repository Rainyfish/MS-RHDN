import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if args.data_test == 'video':
    from videotester import VideoTester
    model = model.Model(args, checkpoint)
    t = VideoTester(args, model, checkpoint)
    t.test()
else:
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        params = list(model.parameters())

        k = 0
        for i in params:
            l = 1
            # print("该层的结构：" + str(list(i.size())))
            for j in i.size():
                l *= j
            # print("该层参数和：" + str(l))
            k = k + l

        checkpoint.write_log("总参数数量和：" + str(k))
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()

