package com.neuralnetwork.core.interfaces;

/**
 * @param <I> the underlying data structure
 *           as well as the input
 */
public interface INeuron<I>
{
    IActivationFunction phi();

    double getWeight(int weight);

    void setWeight(int weight, double newWeight);

    int getNumberOfWeights();

    /**
     * @deprecated ???
     *
     * @return
     */
    I getWeightsWithoutBias();

    I getWeights();

    @Override
    String toString();
}
