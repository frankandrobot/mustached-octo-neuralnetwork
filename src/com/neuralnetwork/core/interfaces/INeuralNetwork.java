package com.neuralnetwork.core.interfaces;

import com.neuralnetwork.convolutional.MNeuron;
import com.neuralnetwork.core.NVector;
import com.neuralnetwork.core.Neuron;
import org.ejml.data.DenseMatrix64F;

public interface INeuralNetwork<I,O,N extends INeuron<?>> extends Iterable<N>
{
    public O constructOutput(I input);

    public O constructInducedLocalField(I input);

    public int getNumberOfNeurons();

    public N getNeuron(int neuron);

    public interface IVectorNeuralNetwork extends INeuralNetwork<NVector,NVector,Neuron> {}

    public interface IMatrixNeuralNetwork extends INeuralNetwork<DenseMatrix64F,DenseMatrix64F,MNeuron>
    {
        public DenseMatrix64F getOutput();
    }
}
