package com.neuralnetwork.core.interfaces;

public interface INeuron<I>
{
    double rawoutput(I input);

    double output(I input);

    double getWeight(int weight);

    int getNumberOfWeights();

    IActivationFunction phi();

    I getWeights();

    void setWeight(int weight, double newWeight);

    @Override
    String toString();
}
