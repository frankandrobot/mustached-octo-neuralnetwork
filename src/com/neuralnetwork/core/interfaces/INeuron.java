package com.neuralnetwork.core.interfaces;

/**
 * @param <I> the underlying data structure
 *           as well as the input
 */
public interface INeuron<I>
{
    double rawoutput(I input);

    double output(I input);

    double getWeight(int weight);

    int getNumberOfWeights();

    IActivationFunction phi();

    I getWeightsWithoutBias();

    void setWeight(int weight, double newWeight);

    @Override
    String toString();
}
