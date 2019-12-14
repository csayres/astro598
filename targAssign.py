import time
from multiprocessing import Pool, cpu_count
import pickle
import numpy
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense

from kaiju import RobotGrid, utils



nHexDia = 7
xCoords, yCoords = utils.hexFromDia(nHexDia)
nPositioners = len(xCoords)
print("using %i positioners"%nPositioners)
nTargets = nPositioners
nProcs = 10  # processors to use for multiprocessing
batchSize = 100  # The number of samples to run through the network before the weights / gradient are updated
epochs = 5  # The number of times to iterate through the complete sample of training data
trainingRatio = 0.9


def getValidAssignments(seed):
    """seed is the random seed with which to initialze the RobotGrid

    return dictionary keyed by positioner id with the coordinates of the
    metrology fiber.  These represent valid (non-collided) xy Fiber positions
    for each robot
    """
    rg = RobotGrid(seed=seed)
    for ii, (xp, yp) in enumerate(zip(xCoords, yCoords)):
        rg.addRobot(robotID=ii, xPos=xp, yPos=yp)
    rg.initGrid()
    # give all robots an initial (radom) target configuration
    for robot in rg.robotDict.values():
        # assigns a robot a target picked uniformly in xy
        # from its patrol annulus
        robot.setXYUniform()
    # decollide any colliding robots so that we have a completely
    # non-colliding target configuration
    rg.decollideGrid()
    targetPos = {}
    for robot in rg.robotDict.values():
        targetPos[robot.id] = robot.metFiberPos[:-1] # xy coord, drop the z
    return targetPos


def generateAssignments(nSeeds):
    p = Pool(nProcs)
    tstart = time.time()
    validAssignments = p.map(getValidAssignments, range(nSeeds))
    tend = time.time() - tstart
    print("took %.2f seconds"%tend)
    p.close()
    pickle.dump(validAssignments, open("validAssign_%i.p"%nSeeds, "wb"))


def target2NN(targetDict, shuffle=True):
    y = numpy.zeros((nPositioners, nTargets)) #n x 2, robot x target
    # shuffle targets
    x = []
    shuffledInds = numpy.arange(nPositioners)
    if shuffle:
        numpy.random.shuffle(shuffledInds)
    for targetInd, robotInd in enumerate(shuffledInds):
        target = targetDict[robotInd]
        x.append(target[0]) # xvalue
        x.append(target[1]) # yvalue
        # x is flattened!
        y[robotInd, targetInd] = 1
    x = numpy.array(x)
    # rows and columns of y sum to 1, total sums to nPositioners
    # print("sum of y", numpy.sum(y, axis=0), numpy.sum(y, axis=1), numpy.sum(y))
    y = y.flatten()  # consider normalizing by something?  sum of the array will be 547
    return x, y


def form2NN(assignFile):
    """Take valid assignments from assignFile

    Format for use with the NN.  Shuffle input targets
    """
    numpy.random.seed(547)

    with open(assignFile, "rb") as f:
        validAssignments = pickle.load(f)

    # generate a big array
    # use reshape (nTargs, 2) to get original array
    X = [] # input n x [x1, y1, x2, y2, ... xn, yn]
    Y = [] # output
    for targetDict in validAssignments:
        x, y = target2NN(targetDict)
        X.append(x)
        Y.append(y)
    X = numpy.array(X)
    Y = numpy.array(Y)
    return X, Y


def runNN(X, Y):
    """X is input array xy coords
    Y is output array, flattened nRobots x nTargets array indexing the answers
    """
    # truncate?
    # X = X[:10000,:]
    # Y = Y[:10000,:]
    # normalize

    nTrials = X.shape[0]
    nInputs = X.shape[1]
    nHidden = int(nInputs*1.5)
    nOutputs = Y.shape[1]
    model = Sequential()
    model.add(
        Dense(nHidden,
              activation="relu",
              input_dim = nInputs,
              ))
    model.add(
        Dense(nOutputs, activation="softmax"))
    model.summary()
    model.compile(loss='categorical_crossentropy',  # See: https://keras.io/losses/
                  optimizer='rmsprop',  # See: https://keras.io/optimizers/
                  metrics=['accuracy']
                  )


    # split the data into training and testing, 75% goes towards training
    split = int(numpy.floor(nTrials*trainingRatio))
    X_train = X[:split, :]
    Y_train = Y[:split, :]
    X_test = X[split:, :]
    Y_test = Y[split:, :]

    history = model.fit(X_train, Y_train,
                        batch_size=batchSize, epochs=epochs,
                        verbose=1, validation_data=(X_test, Y_test))
    model.save("targAssign.h5")


def compareModeled():
    model = load_model('targAssign.h5')
    newSeed = 2000000  # never used
    ii = 0
    for seed in range(newSeed, newSeed+10):
        targDict = getValidAssignments(seed)
        x, yTrue = target2NN(targDict, shuffle=False)
        # import pdb; pdb.set_trace()
        print("xhape", x.shape)
        yFit = model.predict(numpy.array([x]), verbose=1)
        yFit = yFit.reshape(nPositioners, nTargets)
        yTrue = yTrue.reshape(nPositioners, nTargets)

        plt.figure()
        plt.imshow(yFit/numpy.sum(yFit))
        plt.title("NN Model Fit %i"%ii)
        plt.ylabel("Positioner Index")
        plt.xlabel("Target Index")
        plt.savefig("model_%i.png"%ii)
        plt.close()
        ii += 1

    plt.figure()
    plt.imshow(yTrue/numpy.sum(yTrue))
    plt.title("True Assignment")
    plt.ylabel("Positioner Index")
    plt.xlabel("Target Index")
    plt.savefig("true.png")
    plt.close()


if __name__ == "__main__":
    nSeeds = 1000000
    generateAssignments(nSeeds)
    X, Y = form2NN("validAssign_%i.p"%nSeeds)
    runNN(X, Y)
    compareModeled()



