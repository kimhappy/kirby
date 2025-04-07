from typing import List
import torch

from ...protocol._config._general_config import _GeneralConfig
from ...protocol._config._item_config    import _ItemConfig
from ...protocol._config._data           import _Data
from ...protocol.impl   .model_base      import ModelBase
from ...protocol.impl   .loss_base       import LossBase
from ...protocol.impl   .optimizer_base  import OptimizerBase
from ...protocol.impl   .scheduler_base  import SchedulerBase
from ...protocol.impl   .trainer_base    import TrainerBase
from ...protocol.impl   .result          import Result
from ..._util   ._cond                   import _merge_cond

class RNN(TrainerBase):
    def __init__(
        self                           ,
        general_config : _GeneralConfig,
        trainer_config : _ItemConfig   ,
        model          : ModelBase     ,
        train_loss     : LossBase      ,
        vali_loss      : LossBase      ,
        optimizer      : OptimizerBase ,
        scheduler      : SchedulerBase ,
        train_data     : List[_Data]   ,
        vali_data      : List[_Data]   ,
        device         : torch.device):
        TrainerBase.__init__(
            self          ,
            general_config,
            trainer_config,
            model         ,
            train_loss    ,
            vali_loss     ,
            optimizer     ,
            scheduler     ,
            train_data    ,
            vali_data     ,
            device)

        self.batch_size       = trainer_config.batch_size
        self.train_init_frame = trainer_config.train_init_frame
        self.train_frame      = trainer_config.train_frame
        self.vali_frame       = trainer_config.vali_frame
        self.chunk_sec        = trainer_config.chunk_sec

        self.chunk_len = int(self.chunk_sec * self.sample_rate)
        self.tis       = torch.empty(0, self.chunk_len, self.num_cond + 1).to(self.device)
        self.tos       = torch.empty(0, self.chunk_len                   ).to(self.device)
        self.vis       = torch.empty(0,                 self.num_cond + 1).to(self.device)
        self.vos       = torch.empty(0                                   ).to(self.device)

        for data in train_data:
            num_chunks = data.input_audio.shape[ 0 ] // self.chunk_len
            len_cut    = num_chunks                  *  self.chunk_len

            input  = torch.from_numpy(_merge_cond(data.input_audio , data.cond)[ :len_cut ].reshape(num_chunks, self.chunk_len, -1)).to(self.device)
            output = torch.from_numpy(            data.output_audio            [ :len_cut ].reshape(num_chunks, self.chunk_len    )).to(self.device)

            self.tis = torch.cat((self.tis, input ))
            self.tos = torch.cat((self.tos, output))

        for data in vali_data:
            input  = torch.from_numpy(_merge_cond(data.input_audio , data.cond)).to(self.device)
            output = torch.from_numpy(            data.output_audio            ).to(self.device)

            self.vis = torch.cat((self.vis, input ))
            self.vos = torch.cat((self.vos, output))

    def train(self) -> Result:
        self.model.train()

        # Prepare shuffle
        num_chunks     = self.tis.shape[ 0 ]
        num_chunks_cut = num_chunks - (num_chunks % self.batch_size)
        shuffle        = (torch.randperm(num_chunks)[ : num_chunks_cut ] - 1).reshape(-1, self.batch_size)
        ep_loss        = 0

        for batch_idxs in shuffle:
            # Prepare batch
            input_batch  = self.tis[ batch_idxs, :, : ]
            target_batch = self.tos[ batch_idxs, :    ]

            # Reset hidden state
            self.model.reset()
            self.model(input_batch[ :, 0: self.train_init_frame, : ])
            self.model.zero_grad()

            # Training
            batch_loss = 0

            for begin in range(self.train_init_frame, input_batch.shape[ 1 ], self.train_frame):
                # Get frame
                input_frame  = input_batch [ :, begin: begin + self.train_frame, : ]
                target_frame = target_batch[ :, begin: begin + self.train_frame    ]

                # Forward pass
                output_batch = self.model     (input_frame)
                loss         = self.train_loss(output_batch, target_frame)

                # Backpropagation
                loss          .backward ()
                self.optimizer.step     ()
                self.model    .detach   ()
                self.optimizer.zero_grad()
                batch_loss += loss

            # Calculate loss
            num_iter  = (input_batch.shape[ 1 ] - self.train_init_frame) // self.train_frame
            ep_loss  += batch_loss / num_iter

        return Result('training loss', (ep_loss / shuffle.shape[ 0 ]).item())

    def validate(self) -> Result:
        self.model.eval()

        with torch.no_grad():
            # Prepare output
            output     = torch.empty_like(self.vos)
            num_frames = self.vos.shape[ 0 ] // self.vali_frame
            remainder  = self.vos.shape[ 0 ] %  self.vali_frame
            self.model.reset()

            # Process frames
            for l in range(num_frames):
                begin                = l     * self.vali_frame
                end                  = begin + self.vali_frame
                input_chunk          = self.vis[ begin: end ]
                output_chunk         = self.model(input_chunk)
                output[ begin: end ] = output_chunk
                self.model.detach()

            # Process remainder
            if remainder != 0:
                begin                = num_frames * self.vali_frame
                end                  = begin + remainder
                input_chunk          = self.vis[ begin: end ]
                output_chunk         = self.model(input_chunk)
                output[ begin: end ] = output_chunk
                self.model.detach()

            # Calculate loss
            loss = self.vali_loss(output, self.vos)

        self.scheduler.step(loss)

        return Result('validation loss', loss.item())
