package com.neuralnetwork.nn.backprop;

import com.neuralnetwork.core.ActivationFunctions;
import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.neuron.Neuron;
import com.neuralnetwork.nn.MultiLayerNN;
import com.neuralnetwork.nn.MultiLayerNNBuilder;
import com.neuralnetwork.nn.layer.NNLayerBuilder;

/**
 * This neural network solves the XOR problem
 */
public class XORNetwork extends MultiLayerNN
{
    static IActivationFunction.IDifferentiableFunction phi =
            new ActivationFunctions.ThresholdFunction();

    public XORNetwork() throws Exception
    {
        super(new MultiLayerNNBuilder()
                        .setLayers(
                                new NNLayerBuilder()
                                        .setNeurons(new Neuron(phi, -1.5,1.0,1.0))
                                        .setNeurons(new Neuron(phi, -0.5, 1.0, 1.0))
                                        .build(),
                                new NNLayerBuilder()
                                        .setNeurons(new Neuron(phi, -0.5,-2.0,1.0))
                                        .build()
                                )
        );
    }
}
