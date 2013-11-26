package com.neuralnetwork.xor;

import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class TwoLayerNetworkTest
{
    @Test
    public void testSimpleOneLayer()
    {
        TwoLayerNetwork network = new TwoLayerNetwork(0.9, 0.04);

        IActivationFunction.IDifferentiableFunction phi = new IActivationFunction.SigmoidUnityFunction();
        network.setGlobalActivationFunction(phi);

        SingleLayorNeuralNetwork layer = new SingleLayorNeuralNetwork();
        layer.setNeurons(new Neuron(phi, 0.25, 0.75, 0.5));
        network.setFirstLayer(layer);

        NVector output = network.output(new NVector(-1, 2));
        assertThat(output.toString(), is("[0.851953]"));
    }

    //f(s,x) := 1/(1+exp(-s*x));
    //g(s,y):=float(at(diff(f(s,x),x),x=y));
    //i:[-1,2];
    //w:[0.25, 0.75, 0.5];
    //f(1, i[1]*w[1] + i[2]*w[2] + w[3]);
}
