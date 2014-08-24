package com.neuralnetwork.nn.backprop;

import com.neuralnetwork.core.Example;
import com.neuralnetwork.core.interfaces.INeuralNetwork;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class ErrorFunction
{
    public double error;

    public double calculate(INeuralNetwork<double[]> network, Example[] examples)
    {
        error = 0;

        for(Example example:examples)
        {
            double[] output = network.generateOutput(example.input);

            for(int i=0; i<output.length; ++i)
            {
                double term = example.expected[i+1] - output[i];

                error += term * term;
            }
        }

        return error;
    }

    public ErrorFunction setTerm(double[] expected, double[] actual)
    {
        assertThat(expected.length, is(actual.length));

        for(int i=0; i<expected.length; ++i)
        {
            double term = expected[i] - actual[i];

            error += term * term;
        }

        return this;
    }

    public ErrorFunction reset()
    {
        error = 0;

        return this;
    }
}
