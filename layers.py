import torch
import torch.nn as nn
import torch.nn.functional as F

class nconv(nn.Module):
    """
    Düğüm Evrişimi (Node Convolution)
    Sensör özelliklerini (x) komşuluk matrisi (A) ile çarparak mekansal bilgi aktarımını sağlar.
    """
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # x boyutu: (batch_size, channels, num_nodes, sequence_length)
        # A boyutu: (num_nodes, num_nodes)
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class linear(nn.Module):
    """
    1x1 Evrişim Katmanı
    Öznitelik boyutlarını dönüştürmek (kanal sayısını ayarlamak) için kullanılır.
    """
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    """
    Grafik Evrişim Ağı (Graph Convolutional Network)
    Hem sabit hem de uyarlanabilir komşuluk matrislerini alıp yayılım (diffusion) işlemini gerçekleştirir.
    """
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h