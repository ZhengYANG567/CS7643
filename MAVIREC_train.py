# This train function is modified based on the train function provided by CircuitNet https://github.com/circuitnet/CircuitNet
def train():
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))

    if not os.path.exists(arg_dict['save_path']):
        os.makedirs(arg_dict['save_path'])
    with open(os.path.join(arg_dict['save_path'],  'arg.json'), 'wt') as f:
      json.dump(arg_dict, f, indent=4)

    arg_dict['ann_file'] = arg_dict['ann_file_train']
    arg_dict['test_mode'] = False 

    print('===> Loading datasets')
    # Initialize dataset
    dataset = build_dataset(arg_dict)

    print('===> Building model')
    # Initialize model parameters
    model = build_model(arg_dict)
    if not arg_dict['cpu']:
        model = model.cuda()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load dataset
    folder_path = "/home/hice1/zyang684/scratch"
    _all_dataset = data_loader(os.path.join(folder_path, "feature"), os.path.join(folder_path, "label"))
    _files = _all_dataset.file_names
    train_files, valid_files = train_test_split(_files, test_size=.2)
    valid_files, test_files = train_test_split(valid_files, test_size=.5)
    train_dataset = deepcopy(_all_dataset)
    train_dataset.file_names = train_files
    valid_dataset = deepcopy(_all_dataset)
    valid_dataset.file_names = valid_files
    test_dataset = deepcopy(_all_dataset)
    test_dataset.file_names = test_files
    
    logging.info(f"{len(train_dataset)} images for train, {len(valid_dataset)} for validation, and {len(test_dataset)} for testing")
    train_loader = train_dataset.get_torch_loader()
    valid_loader = valid_dataset.get_torch_loader()
    test_loader  = test_dataset.get_torch_loader()
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # Build loss
    loss = build_loss(arg_dict)

    # Build Optimzer
    optimizer = optim.AdamW(model.parameters(), lr=arg_dict['lr'],  betas=(0.9, 0.999), weight_decay=arg_dict['weight_decay'])

    # Build lr scheduler
    cosine_lr = CosineRestartLr(arg_dict['lr'], [arg_dict['max_iters']], [1], 1e-7)
    cosine_lr.set_init_lr(optimizer)

    epoch_loss = 0
    iter_num = 0
    print_freq = 100
    save_freq = 10000
    n_epochs = 100
    patience = 30
    train_losses = []
    valid_losses = []
    _exp_name = "Baseline0724"
    for feature, label, _ in dataset:        
        if arg_dict['cpu']:
            input, target = feature, label
        else:
            input, target = feature.cuda(), label.cuda()
        print(input.size())
        break
    # training setup 0724
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-5)

    # train_model(model, optimizer, train_loader, valid_loader, n_epochs=n_epochs, patience=patience,
    #         criterion=nn.MSELoss(), train_losses = train_losses, valid_losses = valid_losses,
    #         prefix = _exp_name)
    # model, train_losses, valid_losses = try_load_model(_exp_name)
    # # Write loss to file
    loss_file = os.path.join(arg_dict['save_path'], "loss.txt")
    with open(loss_file, 'w') as f:
        f.write("Loss historyin")

    # while iter_num < arg_dict['max_iters']:
    #     with tqdm(total=print_freq) as bar:
    #         for feature, label, _ in dataset:        
    #             if arg_dict['cpu']:
    #                 input, target = feature, label
    #             else:
    #                 input, target = feature.cuda(), label.cuda()

    #             regular_lr = cosine_lr.get_regular_lr(iter_num)
    #             cosine_lr._set_lr(optimizer, regular_lr)

    #             model.train()
    #             prediction = model(input)

    #             optimizer.zero_grad()
    #             pixel_loss = loss(prediction, target)
    #             train_losses.append(pixel_loss.item())
    #             epoch_loss += pixel_loss.item()
                
    #             pixel_loss.backward()
    #             optimizer.step()

    #             iter_num += 1
    #             # validate
    #             model.eval()
    #             valid_loss = []
    #             for batch in tqdm(valid_loader):
    #                 imgs, labels = batch
    #                 with torch.no_grad():
    #                     imgs = imgs.unsqueeze(1)
    #                     logits = model(imgs.to(device))
    #                 loss_val = criterion(logits, labels.to(device))
    #                 valid_loss.append(loss_val.item())
    #             valid_loss = sum(valid_loss) / len(valid_loss)
    #             valid_losses.append(valid_loss)
                
    #             bar.update(1)

    #             if iter_num % print_freq == 0:
    #                 break

    #     print("===> Iters[{}]({}/{}): Loss: {:.4f}".format(iter_num, iter_num, arg_dict['max_iters'], epoch_loss / print_freq))
    #     with open(loss_file, 'a') as f:
    #         f.write("{} {:.4f}\n".format(iter_num, epoch_loss / print_freq))
    #     if iter_num % save_freq == 0:
    #         checkpoint(model, iter_num, arg_dict['save_path'])
    #     epoch_loss = 0
    while iter_num < arg_dict['max_iters']:
        model.train()
        train_loss = []
        regular_lr = cosine_lr.get_regular_lr(iter_num)
        cosine_lr._set_lr(optimizer, regular_lr)
        for batch in tqdm(train_loader):
            imgs, labels = batch
            imgs = imgs.unsqueeze(1)
            logits = model(imgs.to(device))
            optimizer.zero_grad()
            pixel_loss = loss(logits, labels.to(device))
            train_loss.append(pixel_loss.item())
            pixel_loss.backward()
            optimizer.step()
        iter_num += 1
        train_loss = sum(train_loss)/len(train_loss)
        train_losses.append(train_loss)
        # validate
        model.eval()
        valid_loss = []
        for batch in tqdm(valid_loader):
            imgs, labels = batch
            with torch.no_grad():
                imgs = imgs.unsqueeze(1)
                logits = model(imgs.to(device))
            loss_val = criterion(logits, labels.to(device))
            valid_loss.append(loss_val.item())
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_losses.append(valid_loss)
    plot_losses(train_losses, valid_losses, "./loss.png")



if __name__ == "__main__":
    train()
