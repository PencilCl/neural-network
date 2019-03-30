import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        NeuralNetwork net = new NeuralNetwork(new int[]{15, 15}, new String[]{"sigmoid"}, 0.05f);

        /**
         * = = = = = = = = = = =
         * = X O = = = = = = = =
         * = O X X = = = = = = =
         * = = = = = = = = = = =
         */
        float[][] data = new float[][]{{1,2},{2,2},{1,1},{2,1}};
        float[][] label = new float[][]{{1,0},{0,1},{0,1},{1,0}};

        System.out.println("正在训练...");
        net.setOnEpochUpdateListener((int epoch, float loss) -> {
            System.out.println(String.format("Epoch: %d, Loss: %.5f", epoch, loss));
        });
        net.train(data, label, 10000);
        System.out.println("训练完毕!");

        // 根据训练结果来检验样本数据
        float[][] result = net.test(data);
        for (int i = 0; i < data.length; ++i) {
            System.out.println(Arrays.toString(data[i]) + ":" + Arrays.toString(result[i]));
        }

        // 根据训练结果来预测一条新数据的分类
        float[] x = new float[]{3, 1};
        float[] res = net.test(x);
        System.out.println(Arrays.toString(x) + ":" + Arrays.toString(res));

        // 保存网络
        System.out.println("保存网络" + (NeuralNetwork.saveNetwork(net, "./net") ? "成功!" : "失败!"));
        // 读取网络并测试
        net = NeuralNetwork.readNetwork("./net");
        if (net == null) {
            System.out.println("读取网络失败");
            return;
        }
        res = net.test(x);
        System.out.println(Arrays.toString(x) + ":" + Arrays.toString(res));
    }
}
