package com.neuralnetwork.xor;

import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class TwoLayerNetworkTest
{
    /**
     * Test 1-layer network
     *
     Maxima code:

     f(s,x):=1/(1 + exp(-s*x));
     g(s,x):=at(diff(f(s,y),y),y=x);

     w:[0.25,0.75,0.5];
     i:[-1,2,1];

     v:sum(w[j]*i[j],j,1,3);
     (0.25 - f(1,v)) * g(1,v);

     */
    @Test
    public void testOneLayerNetworkOutput()
    {
        IActivationFunction.IDifferentiableFunction phi = new IActivationFunction.SigmoidUnityFunction();
        SingleLayorNeuralNetwork layer = new SingleLayorNeuralNetwork();
        layer.setNeurons(new Neuron(phi, 0.25, 0.75, 0.5));

        TwoLayerNetwork.Builder builder = new TwoLayerNetwork.Builder();
        builder.setLearningParam(0.9)
               .setMomentumParam(0.04)
               .setGlobalActivationFunction(phi)
               .setFirstLayer(layer);

        TwoLayerNetwork network = new TwoLayerNetwork(builder);

        NVector rslt = network.output(new NVector(-1,2));
        assertThat(rslt.toString(), is("[0.851953]"));
    }

    /**
     * Test 1-layer network
     */
    @Test
    public void testOneLayerBackpropagation()
    {
        IActivationFunction.IDifferentiableFunction phi = new IActivationFunction.SigmoidUnityFunction();
        SingleLayorNeuralNetwork layer = new SingleLayorNeuralNetwork();
        layer.setNeurons(new Neuron(phi, 0.25, 0.75, 0.5));

        TwoLayerNetwork.Builder builder = new TwoLayerNetwork.Builder();
        builder.setLearningParam(0.9)
               .setMomentumParam(0.04)
               .setGlobalActivationFunction(phi)
               .setFirstLayer(layer);

        TwoLayerNetwork network = new TwoLayerNetwork(builder);

        NVector example = new NVector(-1,2);
        NVector expected = new NVector(0.25);

        network.initLayers(example, expected);

        double error = network.backpropagation();

        //check gradients were built correctly
        double e = 0.25 - phi.apply(1.75);
        double phiPrime = phi.derivative(1.75);
        double gradient = e*phiPrime;
        assertThat(network.aExampleLayers[0][0].vGradients.get(0), is(gradient));

        //check weights were updated correctly
        double w1 = 0.25 + 0.9 * gradient * -1.0;
        assertThat(network.aExampleLayers[0][0].layer.aNeurons[0].getWeight(0), is(w1));
    }

    @Test
    public void testNetworkOutput()
    {
        IActivationFunction.IDifferentiableFunction phi = new IActivationFunction.SigmoidUnityFunction();
        SingleLayorNeuralNetwork layer1 = new SingleLayorNeuralNetwork();
        layer1.setNeurons(new Neuron(phi, 0.25, 0.75, 0.5));
        SingleLayorNeuralNetwork layer2 = new SingleLayorNeuralNetwork();
        layer2.setNeurons(new Neuron(phi, 0.10, -0.25));

        TwoLayerNetwork.Builder builder = new TwoLayerNetwork.Builder();
        builder.setLearningParam(0.9)
               .setMomentumParam(0.04)
               .setGlobalActivationFunction(phi)
               .setFirstLayer(layer1)
               .setSecondLayer(layer2);

        TwoLayerNetwork network = new TwoLayerNetwork(builder);

        double vh = 1.75;
        double vo = 0.10 * phi.apply(vh) - 0.25;
        double o = phi.apply(vo);

        NVector rslt = network.output(new NVector(-1,2));
        assertThat(rslt.get(0), is(o));
    }

    /**
     *
     * Tests two-layer backpropagation.
     * Maxima code:

     f(s,x):=1/(1 + exp(-s*x));
     g(s,x):=at(diff(f(s,y),y),y=x);

     wh:[0.25,0.75,0.5];
     ih:[-1,2,1];

     vh:sum(wh[j]*ih[j],j,1,3);
     oh:[f(1,vh), 1];

     wo:[0.10, -0.25];
     vo:sum(wo[j]*oh[j],j,1,2);
     f(1,vo);

     */
    @Test
    public void testNetworkBackprop()
    {
        IActivationFunction.IDifferentiableFunction phi = new IActivationFunction.SigmoidUnityFunction();
        SingleLayorNeuralNetwork layer1 = new SingleLayorNeuralNetwork();
        layer1.setNeurons(new Neuron(phi, 0.25, 0.75, 0.5));
        SingleLayorNeuralNetwork layer2 = new SingleLayorNeuralNetwork();
        layer2.setNeurons(new Neuron(phi, 0.10, -0.35));

        TwoLayerNetwork.Builder builder = new TwoLayerNetwork.Builder();
        builder.setLearningParam(0.9)
               .setMomentumParam(0.04)
               .setGlobalActivationFunction(phi)
               .setFirstLayer(layer1)
               .setSecondLayer(layer2);

        TwoLayerNetwork network = new TwoLayerNetwork(builder);

        NVector example = new NVector(-1,2);
        NVector expected = new NVector(0.15);

        network.initLayers(example, expected);

        double error = network.backpropagation();

        //check that gradients were built correctly
        double vh = 1.75;
        double yh = phi.apply(vh);
        double vo = 0.10 * yh - 0.35;
        double o = phi.apply(vo);

        double outputGradient = (expected.get(0) - o) * phi.derivative(vo);
        assertThat(network.aExampleLayers[0][1].vGradients.get(0), is(outputGradient));

        double hiddenGradient = phi.derivative(vh) * outputGradient * 0.10;
        assertThat(round(network.aExampleLayers[0][0].vGradients.get(0), 7),
                   is(round(hiddenGradient, 7)));

        //check that weights were updated correctly
        double wo1 = 0.10 + 0.9 * outputGradient * phi.apply(vh);
        assertThat(wo1, is(network.aExampleLayers[0][1].layer.aNeurons[0].getWeight(0)));

        double wo2 = -0.35 + 0.9 * outputGradient;
        assertThat(wo2, is(network.aExampleLayers[0][1].layer.aNeurons[0].getWeight(1)));

        double wh1 = 0.25 + 0.9 * hiddenGradient * (example.get(0));
        assertThat(wh1, is(network.aExampleLayers[0][0].layer.aNeurons[0].getWeight(0)));

        Neuron nh = network.aExampleLayers[0][0].layer.aNeurons[0];
        Neuron no = network.aExampleLayers[0][1].layer.aNeurons[0];

        //run back prop again and check weights again
        double wo1prev = wo1;

        vh = example.get(0)*nh.getWeight(0) + example.get(1)*nh.getWeight(1) + nh.getWeight(2);
        yh = phi.apply(vh);
        vo = no.getWeight(0) * yh + no.getWeight(1);
        o = phi.apply(vo);

        network.backpropagation();

//        double actual = network.output(0,0,example).first();
//        assertThat(o, is(actual));

        //check that weights were updated correctly
        outputGradient = (expected.get(0) - o) * phi.derivative(vo);
        wo1 = wo1prev + 0.04 * 0.10 + 0.9 * outputGradient * yh;
        assertThat(wo1, is(no.getWeight(0)));
    }

    private String round(double num, int precision)
    {
        return String.format("%"+precision+"g", num);
    }
}
