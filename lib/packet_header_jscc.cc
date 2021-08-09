/* -*- c++ -*- */
/*
 * Copyright 2021 gr-deep_jscc_test author.
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include <deep_jscc_test/packet_header_jscc.h>
#include <gnuradio/digital/lfsr.h>
#include <iostream>
#include <stdbool.h>
namespace gr {
namespace deep_jscc_test {

int _get_header_len_from_occupied_carriers(const std::vector<std::vector<int>> &occupied_carriers, int n_syms)
{
  int header_len = 0;
  for (int i = 0; i < n_syms; i++)
  {
    header_len += occupied_carriers[i].size();
  }
  return header_len;
}

packet_header_jscc::sptr
packet_header_jscc::make(
    const std::vector<std::vector<int>> &occupied_carriers,
    int n_syms,
    const std::string &len_tag_key,
    const std::string &frame_len_tag_key,
    const std::string &num_tag_key,
    int bits_per_header_sym,
    int bits_per_payload_sym,
    bool scramble_header)
{
  return packet_header_jscc::sptr(
      new packet_header_jscc(
          occupied_carriers, n_syms, len_tag_key, frame_len_tag_key, num_tag_key,
          bits_per_header_sym, bits_per_payload_sym, scramble_header));
}

packet_header_jscc::packet_header_jscc(
    const std::vector<std::vector<int>> &occupied_carriers,
    int n_syms,
    const std::string &len_tag_key,
    const std::string &frame_len_tag_key,
    const std::string &num_tag_key,
    int bits_per_header_sym,
    int bits_per_payload_sym,
    bool scramble_header) : packet_header_default(_get_header_len_from_occupied_carriers(occupied_carriers, n_syms), // this function is a bit weird... seems to depend on the number of occupied carriers (48)
                                                  len_tag_key,
                                                  num_tag_key,
                                                  bits_per_header_sym),

                            Md_header_len(_get_header_len_from_occupied_carriers(occupied_carriers, n_syms)),
                            Md_len_tag_key(pmt::string_to_symbol(len_tag_key)),
                            Md_num_tag_key(num_tag_key.empty() ? pmt::PMT_NIL : pmt::string_to_symbol(num_tag_key)),
                            Md_bits_per_byte(bits_per_header_sym),
                            Md_header_number(0),
                            d_frame_len_tag_key(pmt::string_to_symbol(frame_len_tag_key)),
                            d_occupied_carriers(occupied_carriers),
                            d_bits_per_payload_sym(bits_per_payload_sym),
                            d_scramble_mask(d_header_len, 0)
{

  // Init scrambler mask
  if (scramble_header)
  {
    // These are just random values which already have OK PAPR:
    gr::digital::lfsr shift_reg(0x8a, 0x6f, 7);
    for (int i = 0; i < d_header_len; i++)
    {
      for (int k = 0; k < bits_per_header_sym; k++)
      {
        d_scramble_mask[i] ^= shift_reg.next_bit() << k;
      }
    }
  }
}

packet_header_jscc::~packet_header_jscc()
{
}

void packet_header_jscc::insert_into_header_buffer(unsigned char *out, int &currentOffset, unsigned value_to_insert, int number_of_bits_to_copy)
{

  //using namespace std; // Probably for debug
  //cout << "Number to insert " << value_to_insert << endl;
  for (int i = 0; i < number_of_bits_to_copy && currentOffset < Md_header_len; i += Md_bits_per_byte, currentOffset++)
  {
    out[currentOffset] = (unsigned char)((value_to_insert >> i) & d_mask);
  }
}

unsigned packet_header_jscc::extract_from_header_buffer(std::vector<unsigned char> &in, int &currentOffset, int size_of_field)
{
  unsigned result = 0;
  for (int i = 0; i < size_of_field && currentOffset < Md_header_len; i += Md_bits_per_byte, currentOffset++)
  {
    result |= (((int)in[currentOffset]) & d_mask) << i;
  }
  return result;
}

bool packet_header_jscc::header_formatter(long packet_len, unsigned char *out, const std::vector<tag_t> &tags)
{
  bool new_gop = 0;
  bool first = 0;
  unsigned bw_policy = 0;
  // Find our Image Number
  for (size_t i = 0; i < tags.size(); i++)
  {
    if (pmt::equal(tags[i].key, pmt::intern("new_gop")))
    {
      new_gop = static_cast<bool>(pmt::to_bool(tags[i].value));
    }
    if (pmt::equal(tags[i].key, pmt::intern("first")))
    {
      first = static_cast<bool>(pmt::to_bool(tags[i].value));
    }
    if (pmt::equal(tags[i].key, pmt::intern("bw_policy")))
    {
      bw_policy = static_cast<unsigned int>(pmt::to_long(tags[i].value));
    }
  }

  // bool ret_val = packet_header_default::header_formatter(packet_len, out, tags);
  // TODO check the function of the following block
  packet_len &= 0x0FF;
  Md_crc_impl.reset();
  Md_crc_impl.process_bytes((void const *)&new_gop, 3);
  Md_crc_impl.process_bytes((void const *)&first, 2);
  Md_crc_impl.process_bytes((void const *)&bw_policy, 1);
  // Md_crc_impl.process_bytes((void const *) &packet_len, 1);
  // Md_crc_impl.process_bytes((void const *) &Md_header_number, 1);
  unsigned char crc = Md_crc_impl();
  // if (packet_len!=96){
  //   std::cout<<"Packet length was actially "<<packet_len<<std::endl;
  //   throw std::runtime_error("Unexpected Packet length");
  //   EXIT_FAILURE;
  // }

  memset(out, 0x00, Md_header_len);
  int k = 0; // Position in out

  for (int i = 0; i < 3; i++) // FIXME this is hard coded to give 48 bits, should find a more flexible solution
  {
    //Image_number
    insert_into_header_buffer(out, k, new_gop, 2);
    insert_into_header_buffer(out, k, first, 2);
    insert_into_header_buffer(out, k, new_gop, 12);
    // insert_into_header_buffer(out,k,crc,8);
  }

  //for (int i = 0; i < 48; i++) // Probably for debug
  //{
  //  std::cout << "Out " << i << " is " << (int)out[i] << std::endl;
  //}
  //assert(k == 48);

  // aff3ct::module::Encoder_RS enc(24,48);
  return true;
}

bool packet_header_jscc::header_parser(
    const unsigned char *in,
    std::vector<tag_t> &tags)
{
  //descrable the header
  std::vector<unsigned char> in_descrambled(d_header_len, 0);
  for (int i = 0; i < d_header_len; i++)
  {
    in_descrambled[i] = in[i];
    // std::cout<<"Descrabled"<<i<<in_descrambled[i]<<std::endl;
  }

  // unsigned header_len = 0;
  // unsigned header_num = 0;

  tag_t tagH;

  int k = 0; // Position in "in"

  std::vector<bool> header_new_gop(3);
  std::vector<bool> header_first(3);
  std::vector<unsigned> header_bw_policy(3);

  for (int i = 0; i < 3; i++)
  {
    header_new_gop[i] = extract_from_header_buffer(in_descrambled, k, 2);
    header_first[i] = extract_from_header_buffer(in_descrambled, k, 2);
    header_bw_policy[i] = extract_from_header_buffer(in_descrambled, k, 12);
    // unsigned header_crc = extract_from_header_buffer(in_descrambled,k,8);
  }
  header_new_gop[0] = (header_new_gop[0] & header_new_gop[1]) | (header_new_gop[1] & header_new_gop[2]) | (header_new_gop[0] & header_new_gop[2]);
  header_first[0] = (header_first[0] & header_first[1]) | (header_first[1] & header_first[2]) | (header_first[0] & header_first[2]);
  header_bw_policy[0] = (header_bw_policy[0] & header_bw_policy[1]) | (header_bw_policy[1] & header_bw_policy[2]) | (header_bw_policy[0] & header_bw_policy[2]);
  // Md_crc_impl.reset();
  // Md_crc_impl.process_bytes((void const *) &header_image_num, 2);
  // Md_crc_impl.process_bytes((void const *) &header_chunk_num, 1);
  // // Md_crc_impl.process_bytes((void const *) &packet_len, 1);
  // // Md_crc_impl.process_bytes((void const *) &Md_header_number, 1);
  // unsigned char crc = Md_crc_impl();

  // Fail if CRC f

  // Fail if header is longer than expected
  if (k > Md_header_len)
  {
    return false;
  }

  // int packet_len = std::ceil(96 / d_bits_per_payload_sym); // TODO 48 hardcoded by piers, not sure why, possibly related to # occupied carriers
  // TODO also, d_bits_per_payload_sym == 1, unclear why

  int packet_len = 96; // TODO find a way to not have to hard code this number

  tagH.key = pmt::intern("packet_len"); // TODO this fixes the problem for now, still unclear why the header_formatter returns packet_len=48
  tagH.value = pmt::from_long(packet_len);
  tags.push_back(tagH);

  tagH.key = pmt::intern("new_gop");
  tagH.value = pmt::from_bool(header_new_gop[0]);
  tags.push_back(tagH);

  tagH.key = pmt::intern("first");
  tagH.value = pmt::from_bool(header_first[0]);
  tags.push_back(tagH);

  tagH.key = pmt::intern("bw_policy");
  tagH.value = pmt::from_long(header_bw_policy[0]);
  tags.push_back(tagH);

  // if (Md_num_tag_key == pmt::PMT_NIL)
  // {
  //   k += 12;
  // }
  // else
  // {
  //   for (int i = 0; i < 8 && k < Md_header_len; i += Md_bits_per_byte, k++)
  //   {
  //     header_num |= (((int)in_descrambled[k]) & d_mask) << i;
  //   }
  //   tagH.key = d_num_tag_key;
  //   tagH.value = pmt::from_long(header_num);
  //   tags.push_back(tagH);
  // }
  // if (k >= Md_header_len)
  // {
  //   return true;
  // }

  // int packet_len = 0; //# of complex symbols in this frame
  // for (size_t i = 0; i < tags.size(); i++)
  // {
  //   if (pmt::equal(tags[i].key, Md_len_tag_key))
  //   {
  //     // Convert bytes to complex symbols:
  //     packet_len = std::ceil(pmt::to_long(tags[i].value) * 8 / d_bits_per_payload_sym);
  //     //if (pmt::to_long(tags[i].value) * 8 % d_bits_per_payload_sym)
  //     //{
  //     //  packet_len++;
  //     //}
  //     tags[i].value = pmt::from_long(packet_len);
  //     break;
  //   }
  // }

  // To figure out how many payload OFDM symbols there are in this frame,
  // we need to go through the carrier allocation and count the number of
  // allocated carriers per OFDM symbol.
  // frame_len == # of payload OFDM symbols in this frame
  int frame_len = 0;
  /*size_t*/ k = 0; // position in the carrier allocation map
  int symbols_accounted_for = 0;
  while (symbols_accounted_for < packet_len)
  {
    frame_len++;
    symbols_accounted_for += d_occupied_carriers[k].size();
    k = (k + 1) % d_occupied_carriers.size();
  }
  tag_t tag;
  tag.key = d_frame_len_tag_key;
  tag.value = pmt::from_long(frame_len);
  tags.push_back(tag);

  return true;
}


  } /* namespace deep_jscc_test */
} /* namespace gr */

