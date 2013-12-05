package com.neuralnetwork.xor;

import org.junit.Test;

import java.util.Random;

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
        assertThat(network.aLayers[0].vGradients.get(0), is(gradient));

        //check weights were updated correctly
        double w1 = 0.25 + 0.9 * gradient * -1.0;
        assertThat(network.aLayers[0].layer.aNeurons[0].getWeight(0), is(w1));
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
        final double ETA = 0.9;
        final double ALPHA = 0.04;
        final double[] orig_wh = {0.25, 0.75, 0.5};
        final double[] orig_wo = {0.10, -0.35};

        IActivationFunction.IDifferentiableFunction phi = new IActivationFunction.SigmoidUnityFunction();
        SingleLayorNeuralNetwork layer1 = new SingleLayorNeuralNetwork();
        layer1.setNeurons(new Neuron(phi, orig_wh));
        SingleLayorNeuralNetwork layer2 = new SingleLayorNeuralNetwork();
        layer2.setNeurons(new Neuron(phi, orig_wo));

        TwoLayerNetwork.Builder builder = new TwoLayerNetwork.Builder();
        builder.setLearningParam(ETA)
               .setMomentumParam(ALPHA)
               .setGlobalActivationFunction(phi)
               .setFirstLayer(layer1)
               .setSecondLayer(layer2);

        TwoLayerNetwork network = new TwoLayerNetwork(builder);

        NVector example = new NVector(-1,2);
        NVector expected = new NVector(0.15);

        network.initLayers(example, expected);

        double error = network.backpropagation();

        //calculate gradients and weights i.e., perform backprop manually
        double vh = 1.75;
        double yh = phi.apply(vh);
        double vo = orig_wo[0]*yh + orig_wo[1];
        double o = phi.apply(vo);

        double outputGradient = (expected.get(0) - o) * phi.derivative(vo);
        double hiddenGradient = phi.derivative(vh) * outputGradient * orig_wo[0];

        double wo1 = orig_wo[0] + ETA*outputGradient*yh;
        double wo2 = orig_wo[1] + ETA*outputGradient;
        double wh1 = orig_wh[0] + ETA*hiddenGradient*example.get(0);

        final Neuron nh = network.aLayers[0].layer.aNeurons[0];
        final Neuron no = network.aLayers[1].layer.aNeurons[0];

        //check that gradients calculated correctly
        assertThat(network.aLayers[1].vGradients.get(0), is(outputGradient));
        assertThat(round(network.aLayers[0].vGradients.get(0), 7),
                is(round(hiddenGradient, 7)));

        //check that weights were updated correctly
        assertThat(wo1, is(no.getWeight(0)));
        assertThat(wo2, is(no.getWeight(1)));
        assertThat(wh1, is(nh.getWeight(0)));

        //manually re-run back prop and check gradients and weights
        vh = example.get(0)*nh.getWeight(0) + example.get(1)*nh.getWeight(1) + nh.getWeight(2);
        yh = phi.apply(vh);
        vo = no.getWeight(0)*yh + no.getWeight(1);
        o = phi.apply(vo);

        outputGradient = (expected.get(0) - o) * phi.derivative(vo);
        final double wo1prev = wo1;
        wo1 = wo1prev + ALPHA*orig_wo[0] + ETA*outputGradient*yh;

        network.backpropagation();

//        double actual = network.output(0,0,example).first();
//        assertThat(o, is(actual));

        assertThat(wo1, is(no.getWeight(0)));
    }

    private String round(double num, int precision)
    {
        return String.format("%"+precision+"g", num);
    }

    @Test
    public void testStability()
    {
        IActivationFunction.SigmoidUnityFunction phi = new IActivationFunction.SigmoidUnityFunction();

        final long stableSeed = 100012;
        Random r = new Random(stableSeed);

        SingleLayorNeuralNetwork firstLayer = new SingleLayorNeuralNetwork();
        firstLayer.setNeurons(
                new Neuron(phi, r.nextDouble()-0.5, r.nextDouble()-0.5, r.nextDouble()-0.5),
                new Neuron(phi, r.nextDouble()-0.5, r.nextDouble()-0.5, r.nextDouble()-0.5));
        SingleLayorNeuralNetwork secondLayer = new SingleLayorNeuralNetwork();
        secondLayer.setNeurons(new Neuron(phi, r.nextDouble()-0.5, r.nextDouble()-0.5, r.nextDouble()-0.5));

        TwoLayerNetwork.Builder builder = new TwoLayerNetwork.Builder()
                .setMomentumParam(0.05)
                .setLearningParam(0.9)
                .setGlobalActivationFunction(phi)
                .setFirstLayer(firstLayer)
                .setSecondLayer(secondLayer);

        TwoLayerNetwork network = new TwoLayerNetwork(builder);

        final NVector input = new NVector(0.5, 0.2);
        final NVector expected = new NVector(0.8);
        final double errorTol = 0.00001;
        network.backpropagation(
                errorTol,
                input, expected);

        NVector rslt = network.output(0,0,input);
        System.out.format("%nrslt - expected = %s - %s = %s%n", rslt, expected, rslt.subtract(expected));
        System.out.println("error: "+rslt.subtract(expected).dotProduct());
        assertThat(rslt.subtract(expected).dotProduct() < errorTol, is(true));
    }
}
