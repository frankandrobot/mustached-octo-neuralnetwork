package com.neuralnetwork.xor;

import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class TwoLayerNetworkTest
{
    @Test
    public void testSimpleNetworkOutput()
    {
        IActivationFunction.IDifferentiableFunction phi = new IActivationFunction.SigmoidUnityFunction();
        SingleLayorNeuralNetwork layer = new SingleLayorNeuralNetwork();
        layer.setNeurons(new Neuron(phi, 0.25, 0.75, 0.5));

        TwoLayerNetwork.Builder builder = new TwoLayerNetwork.Builder();
        builder.setMomentumParam(0.9)
               .setLearningParam(0.04)
               .setGlobalActivationFunction(phi)
               .setFirstLayer(layer);

        TwoLayerNetwork network = new TwoLayerNetwork(builder);

        NVector rslt = network.output(new NVector(-1,2));
        assertThat(rslt.toString(), is("[0.851953]"));
    }

    @Test
    public void testSimpleBackpropagation1()
    {
        IActivationFunction.IDifferentiableFunction phi = new IActivationFunction.SigmoidUnityFunction();
        SingleLayorNeuralNetwork layer = new SingleLayorNeuralNetwork();
        layer.setNeurons(new Neuron(phi, 0.25, 0.75, 0.5));

        TwoLayerNetwork.Builder builder = new TwoLayerNetwork.Builder();
        builder.setMomentumParam(0.9)
               .setLearningParam(0.04)
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

        //check
    }
}
