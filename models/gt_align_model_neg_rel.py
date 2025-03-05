from .gt_align_model import *

class GtInvAlignTransformer_neg_rel(GtInvAlignTransformer):
    def forward(self, data_dict):

        out = {}
        
        vid_inp = data_dict['feats'].type(torch.FloatTensor).cuda() # ( B, T, C )
        vid_inp = vid_inp.permute([0,2,1]) # (B, C, T)    
        
        text_inp = data_dict['txt'] # ( B, )
        
        target_vector = data_dict['gt_vec'].type(torch.FloatTensor).cuda()

        txt_emb = self.text_model(text_inp) # (B, 768, T_txt, 1) 
        
        txt_emb = txt_emb.type(torch.FloatTensor).cuda()

        assert self.opts.n_dec_layers >= 2, 'Should have more than one decoder layer with mulitple decoder inputs'
        txt_emb = txt_emb.squeeze(-1) # (B, 768, T_txt)

        # -- project text  
        txt_emb = self.input_proj_txt(txt_emb) * math.sqrt(self.opts.d_model) # (B, C, 1)
        txt_emb = txt_emb.permute([2,0,1]) # (1, B, C)

        # -- positional encoding
        if self.opts.positional_encoding_text:
            txt_emb = self.positional_enc(txt_emb)

        # -- project vid  
        vid_emb = self.input_proj_vid(vid_inp) # (B, C, T)
        vid_emb = vid_emb.permute([2,0,1]) # (T, B, C)       

        if self.opts.concatenate_prior:
            if 'pr_expand_vec' in data_dict:
                pr_vec = data_dict['pr_expand_vec'].type(torch.FloatTensor).cuda()
            else:
                pr_vec = data_dict['pr_vec'].type(torch.FloatTensor).cuda()
            pr_vec = pr_vec.cuda()
            ref_inp = self.ref_vec_embedding(pr_vec) 
            ref_inp = ref_inp.permute([1,0,2])
            vid_emb = torch.cat((vid_emb,ref_inp),2)       
                    
        if self.opts.concatenate_prior: 
            vid_emb = self.reproject_concatenate(vid_emb)

        vid_emb = vid_emb * math.sqrt(self.opts.d_model)
        vid_emb = self.positional_enc(vid_emb)

        # inverted transformer
        # trf_output = self.transformer(src=txt_emb, tgt=vid_emb) # (B, C)
        memory = self.transformer.encoder(txt_emb)
        trf_output = self.transformer.decoder(vid_emb, memory) # (T, B, C)

        hs = trf_output

        hs = hs.permute([1,0,2])

        lin_layer = self.fc(hs).squeeze(-1)

        target_vector = target_vector.squeeze(-1).cuda()
        loss = self.loss(lin_layer, target_vector)

        out['loss'] = loss

        if getattr(self.opts, 'neg_lambda', 0) > 0:
            neg_vid_emb = torch.cat([vid_emb[:,1:], vid_emb[:,:1]], dim=1)
            neg_trf_output = self.transformer.decoder(neg_vid_emb, memory)
            neg_hs = neg_trf_output.permute([1,0,2]) # B, T, C
            neg_lin_layer = self.fc(neg_hs).squeeze(-1)
            out['neg_loss'] = self.loss(neg_lin_layer, torch.zeros_like(target_vector))

            if getattr(self.opts, 'rel_lambda', 0) > 0:
                target_vector_cat = torch.cat([target_vector, torch.zeros_like(target_vector)], dim=-1)
                lin_layer_cat = torch.cat([lin_layer, neg_lin_layer], dim=-1)
                # numerical stability
                logits_cat = lin_layer_cat - torch.max(lin_layer_cat, dim=1, keepdim=True)[0]
                # softmax
                exp_logits_cat = torch.exp(logits_cat)
                log_prob_cat = logits_cat - torch.log(exp_logits_cat.sum(1, keepdim=True) + 1e-6)
                mean_log_prob_cat = (target_vector_cat * log_prob_cat).sum(1) / (target_vector_cat.sum(1) + 1e-6)
                out['rel_loss'] = - mean_log_prob_cat
        else:
            assert getattr(self.opts, 'rel_lambda', 0) == 0

        outputs = torch.sigmoid(lin_layer.detach())

        out['preds'] = outputs.cpu().numpy()
        if self.opts.concatenate_prior: 
            out['pr_vec'] = pr_vec.cpu().numpy()
        else: 
            out['pr_vec'] = target_vector.cpu().numpy()
        out['gt_vec'] = target_vector.cpu().numpy()

        # -- calc f1 metrics 
        if target_vector.sum().item() > 0:
            correct, tp, fp, fn, total = calc_align_metrics(outputs.round().detach().cpu().numpy(),
                                                        target_vector.detach().cpu().numpy(),
                                                        inp_fmt='vector',
                                                        BACKGROUND_LABEL = 0,
                                                        overlaps = (0.5,)
                                                        )
            out.update( {'correct': correct, 'tp': tp, 'fp': fp, 'fn': fn, 'total_frames': total} )

        if self.opts.concatenate_prior: 
            if target_vector.sum().item() > 0:
                base_outputs = data_dict['pr_vec'].detach().int().squeeze(-1).cpu().numpy()
                correct_b, tp_b, fp_b, fn_b, total_b = calc_align_metrics(pr_vec.detach().cpu().numpy(),
                                                                target_vector.detach().cpu().numpy(),
                                                                inp_fmt='vector',
                                                                BACKGROUND_LABEL = 0,
                                                                overlaps = (0.5,)
                                                                )

                out.update( {'correct_b': correct_b, 'tp_b': tp_b, 'fp_b': fp_b, 'fn_b': fn_b, 'total_frames_b': total_b} )

        return out
