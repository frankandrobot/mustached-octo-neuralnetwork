package com.neuralnetwork.xor;

import com.neuralnetwork.core.ActivationFunctions;
import com.neuralnetwork.core.NVector;
import com.neuralnetwork.core.Neuron;
import com.neuralnetwork.core.SingleLayerNeuralNetwork;

import java.util.Random;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class XORBackPropagationTest
{
    @org.junit.Test
    public void testAllErrorsDecrease() throws Exception
    {
        ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();

        final long stableSeed = 99991000;
        Random r = new Random(stableSeed);

        SingleLayerNeuralNetwork firstLayer = new SingleLayerNeuralNetwork();
        firstLayer.setNeurons(
                new Neuron(phi, r.nextGaussian(), r.nextGaussian(), r.nextGaussian()),
                new Neuron(phi, r.nextGaussian(), r.nextGaussian(), r.nextGaussian()),
                new Neuron(phi, r.nextGaussian(), r.nextGaussian(), r.nextGaussian())
        );
        SingleLayerNeuralNetwork secondLayer = new SingleLayerNeuralNetwork();
        secondLayer.setNeurons(new Neuron(phi,
                r.nextGaussian(),
                r.nextGaussian(),
                r.nextGaussian(),
                r.nextGaussian()));

        TwoLayerNetwork.Builder builder = new TwoLayerNetwork.Builder()
                .setMomentumParam(0.00001)
                .setLearningParam(0.1)
                .setGlobalActivationFunction(phi)
                .setFirstLayer(firstLayer)
                .setSecondLayer(secondLayer)
                .setIterations(15000);

        TwoLayerNetwork network = new TwoLayerNetwork(builder);

        final double tolerance = 0.0001;

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
