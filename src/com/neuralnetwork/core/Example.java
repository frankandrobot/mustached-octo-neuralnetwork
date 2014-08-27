package com.neuralnetwork.core;

public class Example
{
    /**
     * input[0] = +1 for the bias
     */
    public double[] input;

    /**
     * each neuron corresponds to an element in array
     * except expected[0] = +1 for the bias
     */
    public double[] expected;

    public Example setInput(double... input)
    {
        this.input = input;

        return this;
    }

    public Example setExpected(double... expected)
    {
        this.expected = expected;

        return this;
    }
}
