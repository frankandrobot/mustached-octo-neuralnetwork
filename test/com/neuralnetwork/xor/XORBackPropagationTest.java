package com.neuralnetwork.xor;

import java.util.Random;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class XORBackPropagationTest
{
    @org.junit.Test
    public void testAllErrorsDecrease() throws Exception
    {
        IActivationFunction.SigmoidUnityFunction phi = new IActivationFunction.SigmoidUnityFunction();

        final long stableSeed = 99991000;
        Random r = new Random(stableSeed);

        SingleLayorNeuralNetwork firstLayer = new SingleLayorNeuralNetwork();
        firstLayer.setNeurons(
                new Neuron(phi, r.nextGaussian(), r.nextGaussian(), r.nextGaussian()),
                new Neuron(phi, r.nextGaussian(), r.nextGaussian(), r.nextGaussian()),
                new Neuron(phi, r.nextGaussian(), r.nextGaussian(), r.nextGaussian())
        );
        SingleLayorNeuralNetwork secondLayer = new SingleLayorNeuralNetwork();
        secondLayer.setNeurons(new Neuron(phi,
                r.nextGaussian(),
                r.nextGaussian(),
                r.nextGaussian(),
                r.nextGaussian()));

        TwoLayerNetwork.Builder builder = new TwoLayerNetwork.Builder()
                .setMomentumParam(0.9)
                .setLearningParam(0.01)
                .setGlobalActivationFunction(phi)
                .setFirstLayer(firstLayer)
                .setSecondLayer(secondLayer);

        TwoLayerNetwork network = new TwoLayerNetwork(builder);

        final double tolerance = 0.1;

        final NVector input1 = new NVector(0, 0);
        final NVector expected1 = new NVector(0.2);
        final NVector input4 = new NVector(0, 1);
        final NVector expected4 = new NVector(0.8);

        network.backpropagation(tolerance,
                input1, expected1,
                new NVector(1, 0), new NVector(0.8),
                new NVector(1, 1), new NVector(0.2),
                input4, expected4);

        NVector rslt4 = network.output(0,0,input4);
        NVector diff4 = rslt4.subtract(expected4);

        System.out.format("%nrslt - expected = %s - %s = %s%n", rslt4, expected4, diff4);
        System.out.println("error: "+diff4.dotProduct());
        assertThat(diff4.dotProduct() < tolerance, is(true));
    }
}
