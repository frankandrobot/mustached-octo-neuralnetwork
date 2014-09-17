package com.neuralnetwork.cnn;

import com.neuralnetwork.core.interfaces.IMatrixNeuralLayer;

import java.util.ArrayList;
import java.util.HashSet;

public class CNNBuilder {

    ArrayList<CNNConnections> layers = new ArrayList<CNNConnections>();

    private int len = 0;

    public CNNBuilder setLayer(CNNConnection... layers)
    {
        if (this.layers.get(len) == null) this.layers.add(new CNNConnections());

        CNNConnections conns = this.layers.get(len++);

        for(CNNConnection conn:layers)
            conns.connectionList.add(conn);

        return this;
    }

    public CNN build() throws Exception
    {
        validate();

        return null;
    }

    private void validate() throws Exception {
        for(int i=0; i<layers.size()-1; i++)
        {
            CNNConnections cur = layers.get(i);
            CNNConnections next = layers.get(i+1);

            HashSet<IMatrixNeuralLayer> curMaps = getMapsAtLayer(i);

            for(CNNConnection conn:next.connectionList)
                for(IMatrixNeuralLayer prevMap:conn.next)
                {
                    if (!curMaps.contains(prevMap))
                        throw new Exception("Maps must be connected to maps in previous layer");
                }
        }
    }

    private HashSet<IMatrixNeuralLayer> getMapsAtLayer(int layer)
    {
        HashSet<IMatrixNeuralLayer> maps = new HashSet<IMatrixNeuralLayer>();

        for(CNNConnection conn:layers.get(layer).connectionList)
        {
            maps.add(conn.current);
        }

        return maps;
    }
}
