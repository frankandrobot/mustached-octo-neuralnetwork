package com.neuralnetwork.xor;

import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class TwoLayerNetworkTest
{
    /**
     * Test 1-layer network
     *

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
        double vo = 0.10 * phi.apply(1.75) - 0.25;
        double o = phi.apply(vo);

        NVector rslt = network.output(new NVector(-1,2));
        assertThat(rslt.get(0), is(o));
    }
}
