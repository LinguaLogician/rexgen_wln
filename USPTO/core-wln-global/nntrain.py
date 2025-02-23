import tensorflow as tf
from utils.nn import linearND, linear
from mol_graph import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list as _s2g
from models import gated_convnet, rcnn_wl_last
from ioutils import get_all_batch, get_feature_batch, INVALID_BOND
import math, sys, random
from collections import Counter
from optparse import OptionParser
from functools import partial
import threading
from multiprocessing import Queue

NK = 20
NK0 = 10
print("Imported TensorFlow:", tf.__version__)
print("tf.config exists:", hasattr(tf, "config"))
tf.config.set_soft_device_placement(True)
print("Soft device placement enabled.")
# Parse command-line options
parser = OptionParser()
parser.add_option("-t", "--train", dest="train_path", default="/home/liangtao/Development/ChemistrySpace/rexgen_wln/USPTO/data/train.txt")
parser.add_option("-m", "--save_dir", dest="save_path", default="/home/liangtao/Development/ChemistrySpace/rexgen_wln/USPTO/checkpoints")
parser.add_option("-b", "--batch", dest="batch_size", default=20)
parser.add_option("-w", "--hidden", dest="hidden_size", default=100)
parser.add_option("-d", "--depth", dest="depth", default=1)
parser.add_option("-l", "--max_norm", dest="max_norm", default=5.0)
opts, args = parser.parse_args()

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
depth = int(opts.depth)
max_norm = float(opts.max_norm)

# Use the modified smiles2graph function with AtomMapNumber
smiles2graph_batch = partial(_s2g, idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1)

# TensorFlow 2.x setup
tf.config.set_soft_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define input placeholders
_input_atom = tf.keras.Input(shape=(None, adim), dtype=tf.float32, batch_size=batch_size, name="input_atom")
_input_bond = tf.keras.Input(shape=(None, bdim), dtype=tf.float32, batch_size=batch_size, name="input_bond")
_atom_graph = tf.keras.Input(shape=(None, max_nb, 2), dtype=tf.int32, batch_size=batch_size, name="atom_graph")
_bond_graph = tf.keras.Input(shape=(None, max_nb, 2), dtype=tf.int32, batch_size=batch_size, name="bond_graph")
_num_nbs = tf.keras.Input(shape=(None,), dtype=tf.int32, batch_size=batch_size, name="num_nbs")
_node_mask = tf.keras.Input(shape=(None,), dtype=tf.float32, batch_size=batch_size, name="node_mask")
_label = tf.keras.Input(shape=(None,), dtype=tf.int32, batch_size=batch_size, name="label")
# _binary = tf.keras.Input(shape=(None, None, bdim), dtype=tf.float32, batch_size=batch_size, name="binary")

# _binary = tf.keras.Input(shape=(None, None, 10), dtype=tf.float32, batch_size=batch_size, name="binary")
_binary = tf.keras.Input(shape=(None, None, 10), dtype=tf.float32, batch_size=batch_size, name="binary")
# Model definition
graph_inputs = (_input_atom, _input_bond, _atom_graph, _bond_graph, _num_nbs, _node_mask)
atom_hiddens, _ = rcnn_wl_last(graph_inputs, batch_size=batch_size, hidden_size=hidden_size, depth=depth)

atom_hiddens1 = tf.reshape(atom_hiddens, [batch_size, 1, -1, hidden_size])
atom_hiddens2 = tf.reshape(atom_hiddens, [batch_size, -1, 1, hidden_size])
atom_pair = atom_hiddens1 + atom_hiddens2

att_hidden = tf.nn.relu(
    linearND(atom_pair, hidden_size, scope="att_atom_feature", init_bias=None) + linearND(_binary, hidden_size,
                                                                                          scope="att_bin_feature"))
att_score = linearND(att_hidden, 1, scope="att_scores")
# att_score = tf.nn.sigmoid(att_score)
# att_context = att_score * atom_hiddens1

att_score = tf.expand_dims(att_score, axis=-1)  # [20, 66, 1]
att_context = att_score * atom_hiddens1  # 现在形状匹配
att_context = tf.reduce_sum(att_context, axis=2)

att_context1 = tf.reshape(att_context, [batch_size, 1, -1, hidden_size])
att_context2 = tf.reshape(att_context, [batch_size, -1, 1, hidden_size])
att_pair = att_context1 + att_context2

pair_hidden = linearND(atom_pair, hidden_size, scope="atom_feature", init_bias=None) + linearND(_binary, hidden_size,
                                                                                                scope="bin_feature",
                                                                                                init_bias=None) + linearND(
    att_pair, hidden_size, scope="ctx_feature")
pair_hidden = tf.nn.relu(pair_hidden)
pair_hidden = tf.reshape(pair_hidden, [batch_size, -1, hidden_size])

score = linearND(pair_hidden, 1, scope="scores")
score = tf.squeeze(score, axis=[2])
bmask = tf.cast(tf.equal(_label, INVALID_BOND), tf.float32) * 10000
flat_score = tf.reshape(score, [-1])
flat_label = tf.reshape(_label, [-1])
bond_mask = tf.cast(tf.not_equal(flat_label, INVALID_BOND), tf.float32)
flat_label = tf.maximum(0, flat_label)

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss = loss_fn(tf.cast(flat_label, tf.float32), flat_score)
loss = tf.reduce_sum(loss * bond_mask)

# Model compilation
model = tf.keras.Model(
    inputs=[_input_atom, _input_bond, _atom_graph, _bond_graph, _num_nbs, _node_mask, _label, _binary],
    outputs=[score, loss])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred)

# Data loading and processing
queue = Queue()


def read_data(path, coord):
    data = []
    with open(path, 'r') as f:
        for line in f:
            r, e = line.strip("\r\n ").split()
            data.append((r, e))

    random.shuffle(data)

    for it in range(0, len(data), batch_size):
        src_batch, edit_batch = [], []
        for i in range(batch_size):
            react, _, p = data[it][0].split('>')
            src_batch.append(react)
            edits = data[it][1]
            edit_batch.append(edits)
            it = (it + 1) % len(data)

        src_tuple = smiles2graph_batch(src_batch)
        cur_bin, cur_label, sp_label = get_all_batch(zip(src_batch, edit_batch))
        # feed_map = {x: y for x, y in
        #             zip([_input_atom, _input_bond, _atom_graph, _bond_graph, _num_nbs, _node_mask], src_tuple)}
        feed_map = {x.name: y for x, y in
                    zip([_input_atom, _input_bond, _atom_graph, _bond_graph, _num_nbs, _node_mask], src_tuple)}
        # feed_map.update({_label: cur_label, _binary: cur_bin})
        feed_map.update({_label.name: cur_label, _binary.name: cur_bin})

        queue.put(feed_map)

    coord.request_stop()


coord = tf.train.Coordinator()
t = threading.Thread(target=read_data, args=(opts.train_path, coord))
t.start()

# Training loop
it, sum_acc, sum_err, sum_gnorm = 0, 0.0, 0.0, 0.0
try:
    while not coord.should_stop():
        feed_map = queue.get()
        _, cur_topk, cur_loss = model.train_on_batch(feed_map)
        it += 1

        # Calculate accuracy
        sp_label = feed_map[_label]
        for i in range(batch_size):
            pre = 0
            for j in range(NK):
                if cur_topk[i, j] in sp_label[i]:
                    pre += 1
            if len(sp_label[i]) == pre:
                sum_err += 1
            pre = 0
            for j in range(NK0):
                if cur_topk[i, j] in sp_label[i]:
                    pre += 1
            if len(sp_label[i]) == pre:
                sum_acc += 1

        if it % 50 == 0:
            print(
                f"Iter: {it}, Acc@10: {sum_acc / (50 * batch_size):.4f}, Acc@20: {sum_err / (50 * batch_size):.4f}, Loss: {cur_loss:.4f}")
            sum_acc, sum_err = 0.0, 0.0

        if it % 10000 == 0:
            model.save_weights(f"{opts.save_path}/model.ckpt-{it}")
            print("Model Saved!")

except Exception as e:
    print(e)
    coord.request_stop(e)
finally:
    model.save_weights(f"{opts.save_path}/model.final")
    coord.request_stop()
    coord.join([t])