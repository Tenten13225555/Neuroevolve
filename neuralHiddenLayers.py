import numpy as np

from utils.dataproc.dataset import load_data, get_data_loaders
from utils.visual.datasplit_class_balance import class_balance

# from tensorflow import keras
from spektral.layers import EdgeConv, GatedGraphConv, MessagePassing, CensNetConv, GeneralConv
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from spektral.layers.pooling import DiffPool
import tensorflow as tf

# ----------------------------------------------------------Load Data-------------------------------------------------

# Fixed configurations, one would mainly change the 'sub' variable
sub = "pt01"
node_feat = "ones"
edge_feat = "ones"
adj = "corr"
normalize = True
link_cutoff = 0.3
classify = "binary"
combined_features = True
comb_node_feat = None
comb_edge_feat = None
seed = 0
self_loops = False
batch_size = 32
val_size = 0.1
test_size = 0.1

# Fixed configuration of combined features; check main for updates
if combined_features and comb_node_feat is None:
    comb_node_feat = ["energy", "band_energy"]
    if combined_features and comb_edge_feat is None:
        comb_edge_feat = ["coh", "phase"]

# test load_data()
data, ch_names, preictal_size, ictal_size, postictal_size = load_data(sub, node_f=node_feat, edge_f=edge_feat,
                                                                      adj_type=adj, normalize=normalize,
                                                                      link_cutoff=link_cutoff, classification=classify,
                                                                      combined_features=combined_features,
                                                                      comb_node_feat=comb_node_feat,
                                                                      comb_edge_feat=comb_edge_feat,
                                                                      self_loops=self_loops
                                                                      )

# test get_data_loaders()
train_loader, val_loader, test_loader = get_data_loaders(data, batch_size, val_size, test_size, seed)
train_indices = train_loader.sampler.__getattribute__("indices")

train_data = []
for batch in train_loader:
    train_data.append(batch)
train_data = np.concatenate(train_data, axis=0)


val_indices = val_loader.sampler.__getattribute__("indices")

val_data = []
for batch in val_loader:
    val_data.append(batch)
val_data = np.concatenate(val_data, axis=0)

test_indices = test_loader.sampler.__getattribute__("indices")

# test class_balance()
class_balance(preictal_size, ictal_size, postictal_size, train_indices, val_indices, savefig=True)


# ---------------------------------------------- Custom Neural Architecture--------------------------------------------------
class edgeConvNet(Model):
    def __init__(self, fltrs_out=64, l2_reg=1e-3, dropout_rate=0.5, classify="binary"):
        super().__init__()
        self.conv1 = EdgeConv(fltrs_out, node_channels=64, edge_channels=32, kernel_network=[32], activation="relu", kernel_regularizer=l2(l2_reg))
        self.conv2 = EdgeConv(fltrs_out, node_channels=64, edge_channels=32, activation="relu", kernel_regularizer=l2(l2_reg),
                              attn_kernel_regularizer=l2(l2_reg), return_attn_coef=True)
        self.diffpool = SortPool(k=10, activation="relu")
        self.fc = Dense(32, "relu", kernel_regularizer=l2(l2_reg))
        self.dropout = Dropout(dropout_rate)
        if classify == "binary":
            self.out = Dense(1, "sigmoid", kernel_regularizer=l2(l2_reg))
        elif classify == "multi":
            self.out = Dense(3, "softmax", kernel_regularizer=l2(l2_reg))

    def call(self, inputs, training):
        A_in, X_in, E_in = inputs

        x = self.conv1([X_in, A_in, E_in])
        x, attn = self.conv2([x, A_in])
        x = self.diffpool(x)
        x = self.fc(x)
        x = self.dropout(x)
        output = self.out(x)


class gatedGraphConvNet(Model):
    def __init__(self, fltrs_out=64, l2_reg=1e-3, dropout_rate=0.5, classify="binary"):
        super().__init__()
        self.conv1 = GatedGraphConv(fltrs_out, node_channels=64, edge_channels=32, kernel_network=[32], activation="relu", kernel_regularizer=l2(l2_reg))
        self.conv2 = GatedGraphConv(fltrs_out, node_channels=64, edge_channels=32, activation="relu", kernel_regularizer=l2(l2_reg),
                                    attn_kernel_regularizer=l2(l2_reg), return_attn_coef=True)
        self.diffpool = DiffPool(k=10, activation="relu")
        self.fc = Dense(32, "relu", kernel_regularizer=l2(l2_reg))
        self.dropout = Dropout(dropout_rate)
        if classify == "binary":
            self.out = Dense(1, "sigmoid", kernel_regularizer=l2(l2_reg))
        elif classify == "multi":
            self.out = Dense(3, "softmax", kernel_regularizer=l2(l2_reg))

    def call(self, inputs, training):
        A_in, X_in, E_in = inputs

        x = self.conv1([X_in, A_in, E_in])
        x, attn = self.conv2([x, A_in])
        x = self.diffpool(x)
        x = self.fc(x)
        x = self.dropout(x)
        output = self.out(x)


class messagePassingNet(Model):
    def __init__(self, fltrs_out=64, l2_reg=1e-3, dropout_rate=0.5, classify="binary"):
        super().__init__()
        self.conv1 = MessagePassing(fltrs_out, node_channels=64, edge_channels=32, kernel_network=[32], activation="relu", kernel_regularizer=l2(l2_reg))
        self.conv2 = MessagePassing(fltrs_out, node_channels=64, edge_channels=32, activation="relu", kernel_regularizer=l2(l2_reg),
                                    attn_kernel_regularizer=l2(l2_reg), return_attn_coef=True)
        self.diffpool = DiffPool(k=10, activation="relu")
        self.fc = Dense(32, "relu", kernel_regularizer=l2(l2_reg))
        self.dropout = Dropout(dropout_rate)
        if classify == "binary":
            self.out = Dense(1, "sigmoid", kernel_regularizer=l2(l2_reg))
        elif classify == "multi":
            self.out = Dense(3, "softmax", kernel_regularizer=l2(l2_reg))

    def call(self, inputs, training):
        A_in, X_in, E_in = inputs

        x = self.conv1([X_in, A_in, E_in])
        x, attn = self.conv2([x, A_in])
        x = self.diffpool(x)
        x = self.fc(x)
        x = self.dropout(x)
        output = self.out(x)


class censNetConvNet(Model):
    def __init__(self, fltrs_out=64, l2_reg=1e-3, dropout_rate=0.5, classify="binary"):
        super().__init__()
        self.conv1 = CensNetConv(node_channels=64, edge_channels=32, kernel_network=[32], activation="relu", kernel_regularizer=l2(l2_reg))
        self.conv2 = CensNetConv(node_channels=64, edge_channels=32, activation="relu", kernel_regularizer=l2(l2_reg),
                                 attn_kernel_regularizer=l2(l2_reg), return_attn_coef=True)
        self.diffpool = DiffPool(k=10, activation="relu")
        self.fc = Dense(32, "relu", kernel_regularizer=l2(l2_reg))
        self.dropout = Dropout(dropout_rate)
        if classify == "binary":
            self.out = Dense(1, "sigmoid", kernel_regularizer=l2(l2_reg))
        elif classify == "multi":
            self.out = Dense(3, "softmax", kernel_regularizer=l2(l2_reg))

    def call(self, inputs, training):
        A_in, X_in, E_in = inputs

        x = self.conv1([X_in, A_in, E_in])
        x, attn = self.conv2([x, A_in])
        x = self.diffpool(x)
        x = self.fc(x)
        x = self.dropout(x)
        output = self.out(x)


class generalConvNet(Model):
    def __init__(self, fltrs_out=64, l2_reg=1e-3, dropout_rate=0.5, classify="binary"):
        super().__init__()
        self.conv1 = GeneralConv(fltrs_out, node_channels=64, edge_channels=32, kernel_network=[32], activation="relu", kernel_regularizer=l2(l2_reg))
        self.conv2 = GeneralConv(fltrs_out, node_channels=64, edge_channels=32, activation="relu", kernel_regularizer=l2(l2_reg),
                                 attn_kernel_regularizer=l2(l2_reg), return_attn_coef=True)
        self.diffpool = DiffPool(k=10, activation="relu")
        self.fc = Dense(32, "relu", kernel_regularizer=l2(l2_reg))
        self.dropout = Dropout(dropout_rate)
        if classify == "binary":
            self.out = Dense(1, "sigmoid", kernel_regularizer=l2(l2_reg))
        elif classify == "multi":
            self.out = Dense(3, "softmax", kernel_regularizer=l2(l2_reg))

    def call(self, inputs, training):
        A_in, X_in, E_in = inputs

        x = self.conv1([X_in, A_in, E_in])
        x, attn = self.conv2([x, A_in])
        x = self.diffpool(x)
        x = self.fc(x)
        x = self.dropout(x)
        output = self.out(x)


# ----------------------------------------------Training Neural Architectures------------------------------------------
# -----------------------------------------------------------Training---------------------------------------------------

# Define hyperparameters
epochs = 100
learning_rate = 1e-4
patience = 20

for epoch in range(epochs):
    # train
    metrics = None
    for b in train_loader:
        inputs, targets = b
        if classify == "multi":
            targets = to_categorical(targets, num_classes=3)
        outs = model.train_on_batch(inputs, targets)

# Define model
model = edgeConvNet(classify=classify)

# Define optimizer and loss function
optimizer = Adam(lr=learning_rate)
if classify == "binary":
    loss_fn = "binary_crossentropy"
elif classify == "multi":
    loss_fn = "categorical_crossentropy"

# Compile model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

# Define early stopping callback
early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)




# Train model
history = model.fit(
    train_loader,
    val_loader=val_data,
    epochs=epochs,
    callbacks=[early_stopping],
)


# Evaluate model on test data
test_loss, test_accuracy = model.evaluate(test_loader)

# Print test accuracy
print("Test accuracy: {:.2f}%".format(test_accuracy * 100))

# randomly sample, an alternative for the crossover function (for speed computation and efficiency and at that point it should be able to capture strong generations)
# Also we can use BRAIN_GREG's current neural architecture as the baseline or use the base neural architectures themselves
# start off with two neural architectures instead
# utilizing light training and then evaluating to investigate relative performance of the neural architectures as compared to just observing the perfomance scale itself.
# Use a single based-pair mutation 
# throwing in some randomness, shouldn sort 
# introducing the chance of mutation, because theres the possibility it may stay the same. When would it or could it be advantageous based on the p-values for the single-based pair mutations 
# do some reading and re-approach the problem 
# Setting a system between the architectures based on the fitness score to dictate probability of mating, or use the fitness to evaluate if the mating will even happen at all.

# We make a gene pool where everyone gets a chance, but from there we make one probalistically 

# Use a z-score and a sigmoid function and will weigh things that are high and low probability. Also, to measure the probability for normalization of results.
# shift it so that it starts at the mean.