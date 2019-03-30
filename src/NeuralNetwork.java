import java.io.*;

public class NeuralNetwork implements Serializable {
    private static final float SQRT6 = (float) Math.sqrt(6);
    private static final int DEFAULT_EPOCH = 1000;

    private static final float MOMENTUM = 0.9f;

    private int[] unitNum;
    private String[] activationFunc;
    private float lr;

    private float[][][] weights;
    transient private float[][] trainData;
    transient private float[][] errors; // 每个 unit 的误差
    transient private float[][] label;
    transient private int epoch;
    private float[][] a;

    transient private float[][][] v;

    private int depth;

    public static boolean saveNetwork(NeuralNetwork net, String filePath) {
        try {
            OutputStream os = new FileOutputStream(filePath);
            ObjectOutputStream oos = new ObjectOutputStream(os);
            oos.writeObject(net);
            oos.close();
            os.close();
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }

    public static NeuralNetwork readNetwork(String filePath) {
        try {
            InputStream is = new FileInputStream(filePath);
            ObjectInputStream ois = new ObjectInputStream(is);
            NeuralNetwork net = (NeuralNetwork) ois.readObject();
            ois.close();
            is.close();

            return net;
        } catch (Exception e) {
            e.printStackTrace();
        }

        return null;
    }

    public NeuralNetwork(int[] hiddenLayerUnitNum, String[] activationFunc, float learningRate) {
        this.activationFunc = activationFunc;
        this.lr = learningRate;
        this.depth = hiddenLayerUnitNum.length + 2;

        this.unitNum = new int[this.depth];
        for (int i = 1; i < depth - 1; ++i) {
            // 隐藏层增加一个偏置单元
            unitNum[i] = hiddenLayerUnitNum[i - 1] + 1;
        }
    }

    public void train(float[][] trainData, float[][] label) {
        train(trainData, label, DEFAULT_EPOCH);
    }

    public void train(float[][] trainData, float[][] label, int epoch) {
        if (trainData.length <= 0 || trainData.length != label.length) return;

        this.trainData = trainData;
        this.label = label;
        this.epoch = epoch;

        this.unitNum[0] = trainData[0].length;
        this.unitNum[this.unitNum.length - 1] = label[0].length;

        initVar();
        train();
    }

    public float[][] test(float[][] testData) {
        float[][] pred = new float[testData.length][];
        for (int i = 0; i < testData.length; ++i) {
            pred[i] = test(testData[i]);
        }
        return pred;
    }

    public float[] test(float[] testData) {
        float[] pred = new float[unitNum[depth - 1]];
        forward(testData);
        System.arraycopy(a[depth - 1], 0, pred, 0, unitNum[depth - 1]);
        return pred;
    }

    /**
     * 初始化变量
     */
    private void initVar() {
        // create weights and velocity
        weights = new float[depth][][];
        v = new float[depth][][];
        for (int i = 0; i < depth - 1; ++i) {
            weights[i] = new float[unitNum[i]][];
            v[i] = new float[unitNum[i]][];
            for (int j = 0; j < unitNum[i]; ++j) {
                if (i + 1 == depth - 1) {
                    weights[i][j] = new float[unitNum[i + 1]];
                    v[i][j] = new float[unitNum[i + 1]];
                } else {
                    // 偏置单元不需要和上一层连接
                    weights[i][j] = new float[unitNum[i + 1] - 1];
                    v[i][j] = new float[unitNum[i + 1] - 1];
                }
            }
        }
        // init weights
        for (int i = 0; i < depth - 1; ++i) {
            float bound = (float) (SQRT6 / Math.sqrt(unitNum[i] + unitNum[i + 1]));
            for (int j = 0; j < unitNum[i]; ++j) {
                for (int k = 0; k < weights[i][j].length; ++k) {
                    weights[i][j][k] = r(bound);
                }
            }
        }

        // create a
        a = new float[depth][];
        for (int i = 0; i < depth; ++i) {
            a[i] = new float[unitNum[i]];
        }
        // 设置偏置单元为 1
        for (int i = 1; i < depth - 1; ++i) {
            a[i][unitNum[i] - 1] = 1;
        }

        // create errors
        errors = new float[depth][];
        for (int i = 1; i < depth; ++i) {
            errors[i] = new float[unitNum[i]];
        }
    }

    private void train() {
        float[][] trainData = this.trainData;
        float[][] label = this.label;

        for (int i = 0; i < epoch; ++i) {
            for (int j = 0; j < trainData.length; ++j) {
                forward(trainData[j]);
                backward(label[j]);
                sgrOptimizeWithMomentum();
            }
        }
    }

    /**
     * 前向传播, 计算网络
     * @param data
     */
    private void forward(float[] data) {
        System.arraycopy(data, 0, a[0], 0, data.length);
        for (int l = 0; l < depth - 1; ++l) {
            for (int n = 0; n < (unitNum[l + 1] - (l == depth - 2 ? 0 : 1)); ++n) {
                a[l + 1][n] = 0;
                for (int m = 0; m < unitNum[l]; ++m) {
                    a[l + 1][n] += weights[l][m][n] * a[l][m];
                }

                a[l + 1][n] = sigmoid(a[l + 1][n]);
            }
        }
    }

    /**
     * 反向传播, 计算误差
     */
    private void backward(float[] label) {
        // 最后一层误差
        int l = depth - 1;
        for (int i = 0; i < unitNum[l]; ++i) {
            errors[l][i] = a[l][i] - label[i];
        }

        // 前几层误差
        float error;
        float theta;
        while (--l > 0) {
            for (int i = 0; i < unitNum[l]; ++i) {
                error = 0;
                for (int j = 0; j < (unitNum[l + 1] - (l + 1 == depth - 1 ? 0 : 1)); ++j) {
                    error += weights[l][i][j] * errors[l + 1][j];
                }
                theta = a[l][i] * (1 - a[l][i]);
                errors[l][i] = error * theta;
            }
        }
    }

    /**
     * SGR (Stochastic gradient descent, 随机梯度下降)
     */
    private void sgrOptimize() {
        for (int l = 0; l < depth - 1; ++l) {
            for (int i = 0; i < unitNum[l]; ++i) {
                for (int j = 0; j < weights[l][i].length; ++j) {
                    weights[l][i][j] -= errors[l + 1][j] * a[l][i] * this.lr;
                }
            }
        }
    }

    /**
     * 带动量项 SGR
     */
    private void sgrOptimizeWithMomentum() {
        for (int l = 0; l < depth - 1; ++l) {
            for (int i = 0; i < unitNum[l]; ++i) {
                for (int j = 0; j < weights[l][i].length; ++j) {
                    v[l][i][j] = MOMENTUM * v[l][i][j] + (1 - MOMENTUM) * a[l][i] * errors[l + 1][j];
                    weights[l][i][j] -=  this.lr * v[l][i][j];
                }
            }
        }
    }

    private float sigmoid(float x) {
        return (float) (1.0 / (1 + Math.exp(-x)));
    }

    /**
     * 生成 [-bound, bound] 内的随机数
     * @param bound
     * @return
     */
    private float r(float bound) {
        return (float) Math.random() * bound * 2 - bound;
    }

}
