package com.neuralnetwork.nn.backprop;

import com.neuralnetwork.core.Example;
import com.neuralnetwork.core.interfaces.INnLayer;
import com.neuralnetwork.nn.NN;

import static org.hamcrest.CoreMatchers.*;
import static org.junit.Assert.assertThat;

public class NNBackpropBuilder {

    Example[] examples;

    NN net;

    double eta = 0.1;
    double gamma = 0.1;

    int iterations = 5000;

    public NNBackpropBuilder setNet(NN net)
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
        assertThat(examples, not(nullValue()));
        assertThat(net, not(nullValue()));

        INnLayer[] layers = net.getLayers();

        for(Example example:examples)
        {
            assertThat(example.input.length, is(layers[0].getInputDim().cols));
            assertThat(example.expected.length, is(layers[layers.length - 1].getOutputDim().cols + 1));
        }
    }
}