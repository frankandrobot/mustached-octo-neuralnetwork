package com.neuralnetwork.xor;

import java.util.Random;

public class XORBackPropagationNetworkVersionA extends TwoLayerNetwork
{
    public XORBackPropagationNetworkVersionA()
    {
        super(new Builder()
                .setLearningParam(0.9)
                .setMomentumParam(0.04)
                .setGlobalActivationFunction(new IActivationFunction.SigmoidUnityFunction())
                .setFirstLayer(new SingleLayorNeuralNetwork())
                .setSecondLayer(new SingleLayorNeuralNetwork()));

        Random r = new Random();

        this.firstLayer.setNeurons(new Neuron(phi, r.nextDouble()-0.5, r.nextDouble()-0.5, r.nextDouble()-0.5),
                                   new Neuron(phi, r.nextDouble()-0.5, r.nextDouble()-0.5, r.nextDouble()-0.5));

        this.secondLayer.setNeurons(new Neuron(phi, r.nextDouble()-0.5, r.nextDouble()-0.5, r.nextDouble()-0.5));
   }
}
