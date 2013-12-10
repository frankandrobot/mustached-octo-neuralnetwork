package com.neuralnetwork.core.interfaces;

public interface IActivationFunction
{
    public double apply(double v);

    public interface IDifferentiableFunction extends IActivationFunction
    {
        public double derivative(double v);
    }
}
