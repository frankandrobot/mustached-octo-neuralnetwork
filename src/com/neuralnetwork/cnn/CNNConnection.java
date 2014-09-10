package com.neuralnetwork.cnn;

import com.neuralnetwork.core.interfaces.IMatrixNeuralLayer;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class CNNConnection
{
    IMatrixNeuralLayer current;
    List<IMatrixNeuralLayer> next;

    public CNNConnection(IMatrixNeuralLayer current, IMatrixNeuralLayer... next)
    {
        this.current = current;
        this.next = new LinkedList<IMatrixNeuralLayer>(Arrays.asList(next));
    }
}
