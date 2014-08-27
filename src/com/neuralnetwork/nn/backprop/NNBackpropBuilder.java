package com.neuralnetwork.nn.backprop;

import com.neuralnetwork.core.Example;
import com.neuralnetwork.core.interfaces.INeuralLayer;
import com.neuralnetwork.nn.MultiLayerNN;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class NNBackpropBuilder {

    Example[] examples;

    MultiLayerNN net;

    double eta;
    double gamma;

    int iterations;

    public NNBackpropBuilder setNet(MultiLayerNN net)
    {
        this.net = net;

        return this;
    }

    public NNBackpropBuilder setExamples(Example... examples)
    {
        this.examples = examples;

        return this;
    }

    public NNBackpropBuilder setEta(double eta)
    {
        this.eta = eta;

        return this;
    }

    public NNBackpropBuilder setGamma(double gamma)
    {
        this.gamma = gamma;

        return this;
    }

    public NNBackpropBuilder setIterations(int iterations)
    {
        this.iterations = iterations;

        return this;
    }

    NNBackprop build()
    {
        validate();

        return new NNBackprop(this);
    }

    private void validate()
    {
        INeuralLayer[] layers = net.getLayers();

        for(Example example:examples)
        {
            assertThat(example.input.length, is(layers[0].getInputDim()));
            assertThat(example.expected.length, is(layers[layers.length - 1].getOutputDim() + 1));
        }
    }
}