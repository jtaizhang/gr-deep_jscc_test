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

#ifndef INCLUDED_DEEP_JSCC_TEST_PACKET_HEADER_JSCC_H
#define INCLUDED_DEEP_JSCC_TEST_PACKET_HEADER_JSCC_H

#include <deep_jscc_test/api.h>
#include <gnuradio/digital/packet_header_default.h>
#include <vector>

namespace gr {
  namespace deep_jscc_test {

    /*!
     * format looks like:
     *
     * \li new_gop   (2 bits): whether it is a new gop (group of pictures)
     * \li first     (2 bits): whether it is the first frame
     * \li bw_policy (12 bits): bw allocation number
     *
     * Instead of duplicating the payload length, we only add it once
     * and use the CRC8 to make sure it's correctly received.
     *
     * \verbatim
         |  0 -- 1 | 2 -- 3 | 4 -- 15   |   * 3 : hard code for 3 times
         | new_gop |  first | bw_policy |
       \endverbatim
     */
    class DEEP_JSCC_TEST_API packet_header_jscc : public gr::digital::packet_header_default
    {
    public:
      typedef boost::shared_ptr<packet_header_default> sptr;

      packet_header_jscc(
          const std::vector<std::vector<int>> &occupied_carriers,
          int n_syms, const std::string &len_tag_key,
          const std::string &frame_len_tag_key,
          const std::string &num_tag_key,
          int bits_per_header_sym,
          int bits_per_payload_sym,
          bool scramble_header);

      ~packet_header_jscc();
      /*!
       * \brief Encodes the header information in the given tags into
       * bits and places them into \p out.
       *
       * \details
       * Uses the following header format:
       *  - Bits 0-1: whether new gop
       *  - Bits 2-3: whether first frame
       *  - Bit 4-15: bw_policy, 1771 cases (11bit sufficient)
       */
  
      bool header_formatter(
          long packet_len,
          unsigned char *out,
          const std::vector<tag_t> &tags);

      bool header_parser(
          const unsigned char *header,
          std::vector<tag_t> &tags);

      static sptr make(
          const std::vector<std::vector<int>> &occupied_carriers,
          int n_syms,
          const std::string &len_tag_key = "packet_len",
          const std::string &frame_len_tag_key = "frame_len",
          const std::string &num_tag_key = "packet_num",
          int bits_per_header_sym = 1,
          int bits_per_payload_sym = 1,
          bool scramble_header = false);

    protected:
      pmt::pmt_t d_frame_len_tag_key;                          //!< Tag key of the additional frame length tag
      const std::vector<std::vector<int>> d_occupied_carriers; //!< Which carriers/symbols carry data
      int d_bits_per_payload_sym;
      std::vector<unsigned char> d_scramble_mask; //!< Bits are xor'd with this before tx'ing

      long Md_header_len;
      pmt::pmt_t Md_len_tag_key;
      pmt::pmt_t Md_num_tag_key;
      int Md_bits_per_byte;
      unsigned Md_header_number;
      unsigned Md_mask;
      boost::crc_optimal<8, 0x07, 0xFF, 0x00, false, false> Md_crc_impl;

      void insert_into_header_buffer(unsigned char *out, int &currentOffset, unsigned value_to_insert, int number_of_bits_to_copy);
      unsigned extract_from_header_buffer(std::vector<unsigned char> &in, int &currentOffset, int size_of_field);
      //private:
    };

  } // namespace deep_jscc_test
} // namespace gr

#endif /* INCLUDED_DEEP_JSCC_TEST_PACKET_HEADER_JSCC_H */

