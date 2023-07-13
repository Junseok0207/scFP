import argparse

def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='baron_mouse', help='baron_mouse, mouse_es, mouse_bladder, zeisel, baron_human')
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--n_runs', type=int, default=3)

    # FP
    parser.add_argument('--k', type=int, default=15)
    parser.add_argument('--iter', type=int, default=40)
    parser.add_argument('--alpha', type=float, default=0.99)

    # setting
    parser.add_argument('--device', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)

    # Preprocessing
    parser.add_argument('--HVG', type=int, default=2000)
    parser.add_argument('--sf', action='store_true', default=True)
    parser.add_argument('--log', action='store_true', default=True)
    parser.add_argument('--normal', action='store_true', default=False)

    return parser.parse_known_args()

def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals

def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        st_ = "{} <- {} / ".format(name, val)
        st += st_

    return st[:-1]

def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in ['device']:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]