package com.neuralnetwork.xor;

import java.util.Random;

public class XORBackPropagationNetworkVersionA extends TwoLayerNetwork
{
    public XORBackPropagationNetworkVersionA()
    {
        super(new Builder()
                .setMomentumParam(0.05)
                .setLearningParam(9.0)
                .setGlobalActivationFunction(new IActivationFunction.SigmoidUnityFunction())
                .setFirstLayer(new SingleLayorNeuralNetwork())
                .setSecondLayer(new SingleLayorNeuralNetwork()));

        Random r = new Random();

        getLayer(0).layer.setNeurons(new Neuron(phi, r.nextGaussian(), r.nextGaussian(), r.nextGaussian()),
                new Neuron(phi, r.nextGaussian(), r.nextGaussian(), r.nextGaussian()));

        getLayer(1).layer.setNeurons(new Neuron(phi, r.nextGaussian(), r.nextGaussian(), r.nextGaussian()));
   }

}
